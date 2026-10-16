"""Support sets built by literal execution: delete one base row, re-run every query that
reads the table, compare answer fingerprints, roll back.

Deletions are drawn from two seeded random streams (tables, then rows), so a run is
reproducible. Candidates are evaluated by a pool of threads, each with its own DuckDB
cursor and transaction; the accepted support set is the first ``n`` successful
candidates in draw order, so the result does not depend on thread timing.

Paper protocol: a row is deletable only if no other row has identical values (the
historical value-match delete skipped duplicates, concern C12); alias views are never
deletable. Corrected protocol: any physical row is deletable; alias views resolve to the
base table they alias.

Revision 3 adds Qirana neighbours (``neighbours="qirana"``), following Deep and Koutris
(SIGMOD 2017), Section 4: the relation is drawn by the table weights (uniformly at random in
Qirana), then either a row update (one tuple; each attribute that is not part of a primary
key or unique constraint is chosen with probability 0.5 and replaced by a value drawn
uniformly from the attribute's active domain) or a swap update (two tuples of the relation
exchange the values of the chosen attributes), in the ratio ``swap_ratio``. A draw that
leaves the database unchanged is rejected. Every draw is made in the calling thread, so
the accepted support set does not depend on the number of threads.
"""
from __future__ import annotations

import hashlib
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field

import duckdb
import numpy as np

from . import stopflag
from .database import fp_sql, search_path
from .util import LOG, quote_ident, sql_str


def _stop_on(e: BaseException) -> None:
    """Ctrl+C stops the run. Running out of memory also stops it here: with several
    threads it would depend on timing, so the support set could depend on thread count."""
    stopflag.check()
    if stopflag.out_of_memory(e):
        raise RuntimeError("out of memory during support sampling. Run the same command again "
                           "with -Threads 2 (or -Serial); it continues where it stopped.") from e


class _AllRows:
    """Marks a table none of whose rows the historical delete could remove."""

    def __contains__(self, item):
        return True


_ALL_ROWS = _AllRows()


def table_catalog(con) -> dict:
    """(schema, name) lowercased -> table type, for every table and view."""
    rows = con.execute("SELECT table_schema, table_name, table_type FROM information_schema.tables").fetchall()
    return {(s.lower(), t.lower()): (s, t, typ) for s, t, typ in rows}


def resolve_table(b, catalog: dict, alias_map: dict, alias_delete: str):
    """Return (schema, name) of the base table behind key ``b``, or None if not deletable."""
    hit = catalog.get((b[0].lower(), b[1].lower()))
    if hit is None:
        return None
    schema, name, typ = hit
    if typ == "BASE TABLE":
        return (schema, name)
    if alias_delete == "resolve":
        real = alias_map.get((b[0].lower(), b[1].lower()))
        if real is not None:
            hit2 = catalog.get((real[0].lower(), real[1].lower()))
            if hit2 is not None and hit2[2] == "BASE TABLE":
                return (hit2[0], hit2[1])
    return None


@dataclass
class SampleResult:
    n: int
    accepted: list                     # (table key, rowid) in support order
    incidence: dict                    # asset -> sorted list of support indices
    changed_fp: dict                   # asset -> {support index: fingerprint} (if recorded)
    draws: int
    skipped: int
    seconds: float
    checkpoint_seconds: dict = field(default_factory=dict)
    # Revision 3 (study B): per evaluated candidate, in draw order, (accepted index or -1,
    # evaluation seconds, queries executed). Empty unless the sampler traces costs.
    cost_trace: list = field(default_factory=list)
    # Revision 3 (Part 7b): support index -> number of queries that failed on that database
    # (only filled when the sampler isolates errors).
    errors: dict = field(default_factory=dict)

    def counts(self, keys, upto=None) -> np.ndarray:
        if upto is None:
            return np.array([len(self.incidence.get(k, ())) for k in keys], float)
        return np.array([sum(1 for i in self.incidence.get(k, ()) if i < upto) for k in keys], float)

    def edges(self, keys) -> list:
        return [frozenset(self.incidence.get(k, ())) for k in keys]


class SamplingStalled(RuntimeError):
    """Support sampling made no progress for a long time (see SupportSampler.sample)."""


class SupportSampler:
    def __init__(self, con, schemas, assets: dict, readers: dict, base_fp: dict,
                 targets: dict, counts: dict, row_mode: str, threads: int = 4,
                 stall_s: float = 1800.0, neighbours: str = "delete", swap_ratio: float = 0.5,
                 donor_mode: str = "distinct", trace_costs: bool = False,
                 isolate_errors: bool = False, error_as_change: bool = False):
        """
        assets:  key -> (owner, duck sql)
        readers: table key -> list of asset keys that read it
        base_fp: key -> fingerprint of the unmodified answer (None = skip)
        targets: table key -> (schema, name) of the deletable base table, or None
        counts:  table key -> row count used for sampling rows
        """
        self.con, self.schemas = con, schemas
        self.assets, self.readers, self.base_fp = assets, readers, base_fp
        self.targets, self.counts = targets, counts
        self.row_mode, self.threads = row_mode, max(1, int(threads))
        self.stall_s = float(stall_s)
        self._local = threading.local()
        self._dups, self._contig = {}, {}
        self._lock = threading.Lock()
        self._cursor_lock = threading.Lock()
        self._cursors = []
        if neighbours not in ("delete", "qirana"):
            raise ValueError("neighbours must be 'delete' or 'qirana'")
        self.neighbours, self.swap_ratio = neighbours, float(swap_ratio)
        if donor_mode not in ("distinct", "frequency"):
            raise ValueError("donor_mode must be 'distinct' or 'frequency'")
        # Revision 3 (study D): "distinct" draws a new value uniformly from the column's
        # distinct values (the approved default); "frequency" copies the value of a row drawn
        # uniformly, so each value is drawn in proportion to how often it occurs.
        self.donor_mode = donor_mode
        self.trace_costs = bool(trace_costs)
        # Revision 3 (Part 7 fix): DuckDB aborts a transaction after any error, so when one
        # query failed on a neighbour, every later query of that table failed too and was
        # counted as unchanged. With isolate_errors the neighbour is applied again in a fresh
        # transaction after an error, so the remaining queries are evaluated. error_as_change
        # counts a query that fails on the neighbour (but not on D) as changed.
        self.isolate_errors = bool(isolate_errors)
        self.error_as_change = bool(error_as_change)
        self._errs = {}                        # candidate key -> failing queries (isolate_errors)
        self._upd = {}                         # table key -> (columns, eligible, domain sizes)

    # ---- cursors ---------------------------------------------------------------------
    def _new_cursor(self):
        with self._cursor_lock:                # creating cursors is not guaranteed thread-safe
            cur = self.con.cursor()
            self._cursors.append(cur)
        try:                                   # never look up Python variables from worker threads
            cur.execute("SET python_enable_replacements = false")
        except Exception:
            pass
        return cur

    def _cur(self):
        cur = getattr(self._local, "cur", None)
        if cur is None:
            cur = self._new_cursor()
            self._local.cur = cur
        return cur

    def close(self):
        """Close every cursor this sampler opened (the database file stays open otherwise)."""
        with self._cursor_lock:
            curs, self._cursors = self._cursors, []
        for cur in curs:
            try:
                cur.close()
            except Exception:
                pass
        self._local = threading.local()

    @staticmethod
    def _rollback(cur):
        try:
            cur.execute("ROLLBACK")
        except Exception:
            pass

    # ---- table facts (computed once, under a lock) ---------------------------------
    def _table_facts(self, tbl):
        with self._lock:
            if tbl not in self._contig:
                name = f"{quote_ident(tbl[0])}.{quote_ident(tbl[1])}"
                cur = self._new_cursor()
                n, mx = cur.execute(f"SELECT count(*), max(rowid) FROM {name}").fetchone()
                contiguous = (n == 0) or (mx == n - 1)
                if self.row_mode == "unique_only":
                    # Equivalent to the historical value-match test: a row is deletable only
                    # if no other row has the same values in every column (NULLs equal).
                    cols = [d[0] for d in cur.execute(f"SELECT * FROM {name} LIMIT 0").description]
                    if any('"' in c for c in cols):
                        # The historical test quoted column names without escaping, so it
                        # failed on every row of such a table.
                        self._dups[tbl] = _ALL_ROWS
                    else:
                        part = ", ".join(quote_ident(c) for c in cols)
                        dup = cur.execute(
                            f"SELECT rowid FROM (SELECT rowid, count(*) OVER (PARTITION BY {part}) AS __c "
                            f"FROM {name}) WHERE __c > 1").fetchall()
                        self._dups[tbl] = {int(r[0]) for r in dup}
                self._contig[tbl] = contiguous
                cur.close()
                with self._cursor_lock:
                    if cur in self._cursors:
                        self._cursors.remove(cur)
            return self._contig[tbl], self._dups.get(tbl, set())

    # ---- one candidate -------------------------------------------------------------
    def _evaluate_traced(self, cand):
        """(result, evaluation seconds, queries executed) of one candidate (study B)."""
        self._local.nq = 0
        t0 = time.perf_counter()
        res = self._evaluate(cand)
        return res, time.perf_counter() - t0, self._local.nq

    def _evaluate(self, cand):
        if self.neighbours == "qirana":
            return self._evaluate_update(cand)
        return self._evaluate_delete(cand)

    def _evaluate_delete(self, cand):
        b, r, track, record = cand
        tbl = self.targets.get(b)
        if tbl is None or r < 0:
            return None
        contiguous, dups = self._table_facts(tbl)
        cur = self._cur()
        name = f"{quote_ident(tbl[0])}.{quote_ident(tbl[1])}"
        if contiguous:
            rid = int(r)
        else:
            got = cur.execute(f"SELECT rowid FROM {name} ORDER BY rowid LIMIT 1 OFFSET {int(r)}").fetchone()
            if got is None:
                return None
            rid = int(got[0])
        if self.row_mode == "unique_only" and rid in dups:
            return None
        t0, attempt = time.perf_counter(), 0
        while True:
            try:
                cur.execute("BEGIN TRANSACTION")
                deleted = cur.execute(f"DELETE FROM {name} WHERE rowid = {rid}").fetchone()[0]
                break
            except duckdb.TransactionException:
                self._rollback(cur)             # another transaction holds this row: wait for it
                stopflag.check()
                if time.perf_counter() - t0 > self.stall_s:
                    raise SamplingStalled(f"row {rid} of {name} stayed locked for "
                                          f"{self.stall_s / 60:.0f} minutes") from None
                time.sleep(0.002 * (1 + attempt % 25))
                attempt += 1
            except Exception as e:
                self._rollback(cur)
                _stop_on(e)
                return None
        try:
            if deleted != 1:
                return None

            def reapply(c):
                self._apply_retry(c, name, lambda: c.execute(f"DELETE FROM {name} WHERE rowid = {rid}"))
            out = self._changed_answers(cur, b, track, record, reapply)
            if getattr(self._local, "nerr", 0):
                self._errs[(b, r)] = self._local.nerr
            return out
        finally:
            self._rollback(cur)

    def _changed_answers(self, cur, b, track, record, reapply=None):
        """Re-run every query that reads table key ``b`` inside the open transaction and
        return {asset: fingerprint or True} for the answers that changed. ``reapply(cur)``
        opens a new transaction holding the same neighbour (used with isolate_errors)."""
        changed = {}
        self._local.nerr = 0
        for k in self.readers.get(b, ()):
            if track is not None and k not in track:
                continue
            base = self.base_fp.get(k)
            if base is None:
                continue
            own, duck = self.assets[k]
            self._local.nq = getattr(self._local, "nq", 0) + 1
            try:
                cur.execute(f"SET search_path={sql_str(search_path(own, self.schemas))}")
                fp = cur.execute(fp_sql(duck)).fetchone()
            except Exception as e:          # the query's own error: unchanged, as historically
                _stop_on(e)
                if self.isolate_errors and reapply is not None:
                    self._local.nerr = getattr(self._local, "nerr", 0) + 1
                    if self.error_as_change:
                        changed[k] = ("error", type(e).__name__) if record else True
                    self._rollback(cur)
                    reapply(cur)
                continue
            if fp != base:
                changed[k] = fp if record else True
        return changed

    # ---- Qirana neighbours: row and swap updates ---------------------------------------
    DOMAIN_SCHEMA = "__qirana_domain"

    def prepare_updates(self, tables):
        """Columns, update-eligible columns and active domains of every table that can be
        drawn. Runs once in the calling thread before any worker starts. The active domain
        of a column is stored, sorted, as a helper table in schema __qirana_domain, so a
        value is fetched by its position."""
        con = self.con
        con.execute(f"CREATE SCHEMA IF NOT EXISTS {quote_ident(self.DOMAIN_SCHEMA)}")
        keys = {}
        try:
            rows = con.execute("SELECT schema_name, table_name, constraint_column_names FROM duckdb_constraints() "
                               "WHERE constraint_type IN ('PRIMARY KEY', 'UNIQUE')").fetchall()
            for sch, tab, cols in rows:
                keys.setdefault((sch.lower(), tab.lower()), set()).update(c.lower() for c in cols)
        except Exception:
            pass
        for b in tables:
            tbl = self.targets.get(b)
            if tbl is None or b in self._upd:
                continue
            name = f"{quote_ident(tbl[0])}.{quote_ident(tbl[1])}"
            cols = [d[0] for d in con.execute(f"SELECT * FROM {name} LIMIT 0").description]
            excl = keys.get((tbl[0].lower(), tbl[1].lower()), set())
            eligible = [c for c in cols if c.lower() not in excl]
            dom = []
            for c in eligible:
                h = hashlib.sha1(f"{tbl[0]}\x00{tbl[1]}\x00{c}".encode()).hexdigest()[:20]
                dname = f"{quote_ident(self.DOMAIN_SCHEMA)}.{quote_ident('d_' + h)}"
                con.execute(f"CREATE TABLE IF NOT EXISTS {dname} AS "
                            f"SELECT DISTINCT {quote_ident(c)} AS v FROM {name} ORDER BY v NULLS LAST")
                dom.append((dname, int(con.execute(f"SELECT count(*) FROM {dname}").fetchone()[0])))
            self._upd[b] = (cols, eligible, dom)

    def draw_update(self, b, rng):
        """One Qirana neighbour of table key ``b`` as a hashable spec, or None.

        ("row", r, cols, donors): set each column in ``cols`` of physical row ``r`` to the
        active-domain value at position ``donors[j]``.
        ("swap", r1, r2, cols): exchange the values of ``cols`` between rows r1 and r2.
        """
        cols, eligible, dom = self._upd.get(b, ([], [], []))
        cnt = self.counts.get(b, 0)
        if not eligible or cnt <= 0:
            return None
        swap = cnt >= 2 and rng.random() < self.swap_ratio
        while True:                            # each attribute with probability 0.5, at least one
            mask = rng.random(len(eligible)) < 0.5
            if mask.any():
                break
        chosen = tuple(int(j) for j in np.flatnonzero(mask))
        if swap:
            r1 = int(rng.integers(cnt))
            r2 = int(rng.integers(cnt - 1))
            r2 = r2 + 1 if r2 >= r1 else r2
            return ("swap", r1, r2, chosen)
        r = int(rng.integers(cnt))
        if self.donor_mode == "frequency":
            donors = tuple(int(rng.integers(cnt)) for _ in chosen)
        else:
            donors = tuple(int(rng.integers(dom[j][1])) if dom[j][1] > 0 else -1 for j in chosen)
        return ("row", r, chosen, donors)

    def _donor_values(self, cur, name, contiguous, dom, eligible, chosen, donors):
        """New values of a row update: the active-domain value at each donor position, or
        (frequency mode) the column value of each donor row."""
        if self.donor_mode == "frequency":
            out = []
            for j, d in zip(chosen, donors):
                rid_d = self._rid(cur, name, contiguous, d)
                out.append(cur.execute(f"SELECT {quote_ident(eligible[j])} FROM {name} WHERE rowid = {rid_d}")
                           .fetchone()[0])
            return out
        return [cur.execute(f"SELECT v FROM {dom[j][0]} WHERE rowid = {int(d)}").fetchone()[0]
                for j, d in zip(chosen, donors)]

    def _rid(self, cur, name, contiguous, r):
        if contiguous:
            return int(r)
        got = cur.execute(f"SELECT rowid FROM {name} ORDER BY rowid LIMIT 1 OFFSET {int(r)}").fetchone()
        return None if got is None else int(got[0])

    @staticmethod
    def _same(a, b):
        if a is b or a == b:
            return True
        try:
            return bool(np.isnan(a) and np.isnan(b))
        except Exception:
            return False

    def _evaluate_update(self, cand):
        b, spec, track, record = cand
        tbl = self.targets.get(b)
        if tbl is None or spec is None:
            return None
        cols, eligible, dom = self._upd[b]
        contiguous, _ = self._table_facts(tbl)
        cur = self._cur()
        name = f"{quote_ident(tbl[0])}.{quote_ident(tbl[1])}"
        if spec[0] == "row":
            _, r, chosen, donors = spec
            if any(d < 0 for d in donors):
                return None
            rid = self._rid(cur, name, contiguous, r)
            if rid is None:
                return None
            names = [eligible[j] for j in chosen]
            sel = ", ".join(quote_ident(c) for c in names)
            old = cur.execute(f"SELECT {sel} FROM {name} WHERE rowid = {rid}").fetchone()
            new = self._donor_values(cur, name, contiguous, dom, eligible, chosen, donors)
            if old is None or all(self._same(o, v) for o, v in zip(old, new)):
                return None                    # the neighbour would equal D
            stmts = [(rid, names, new)]
        else:
            _, r1, r2, chosen = spec
            rid1 = self._rid(cur, name, contiguous, r1)
            rid2 = self._rid(cur, name, contiguous, r2)
            if rid1 is None or rid2 is None:
                return None
            names = [eligible[j] for j in chosen]
            sel = ", ".join(quote_ident(c) for c in names)
            v1 = cur.execute(f"SELECT {sel} FROM {name} WHERE rowid = {rid1}").fetchone()
            v2 = cur.execute(f"SELECT {sel} FROM {name} WHERE rowid = {rid2}").fetchone()
            if v1 is None or v2 is None or all(self._same(a, c) for a, c in zip(v1, v2)):
                return None
            stmts = [(rid1, names, list(v2)), (rid2, names, list(v1))]
        t0, attempt = time.perf_counter(), 0
        while True:
            try:
                cur.execute("BEGIN TRANSACTION")
                ok = True
                for rid, names_, vals in stmts:
                    sets = ", ".join(f"{quote_ident(c)} = ?" for c in names_)
                    n_upd = cur.execute(f"UPDATE {name} SET {sets} WHERE rowid = {rid}", list(vals)).fetchone()[0]
                    ok = ok and n_upd == 1
                break
            except duckdb.TransactionException:
                self._rollback(cur)
                stopflag.check()
                if time.perf_counter() - t0 > self.stall_s:
                    raise SamplingStalled(f"a row of {name} stayed locked for "
                                          f"{self.stall_s / 60:.0f} minutes") from None
                time.sleep(0.002 * (1 + attempt % 25))
                attempt += 1
            except Exception as e:
                self._rollback(cur)
                _stop_on(e)
                return None
        try:
            if not ok:
                return None

            def reapply(c):
                def run():
                    for rid_, names_, vals_ in stmts:
                        sets_ = ", ".join(f"{quote_ident(x)} = ?" for x in names_)
                        c.execute(f"UPDATE {name} SET {sets_} WHERE rowid = {rid_}", list(vals_))
                self._apply_retry(c, name, run)
            out = self._changed_answers(cur, b, track, record, reapply)
            if getattr(self._local, "nerr", 0):
                self._errs[(b, spec)] = self._local.nerr
            return out
        finally:
            self._rollback(cur)

    def _apply_retry(self, cur, name, run):
        """BEGIN a transaction and apply a neighbour, waiting while another transaction
        holds the row (as the first application does)."""
        t0, attempt = time.perf_counter(), 0
        while True:
            try:
                cur.execute("BEGIN TRANSACTION")
                run()
                return
            except duckdb.TransactionException:
                self._rollback(cur)
                stopflag.check()
                if time.perf_counter() - t0 > self.stall_s:
                    raise SamplingStalled(f"a row of {name} stayed locked for "
                                          f"{self.stall_s / 60:.0f} minutes") from None
                time.sleep(0.002 * (1 + attempt % 25))
                attempt += 1

    def diagnose_neighbour(self, b, spec):
        """Every query that reads ``b``, evaluated on one neighbour in reader order, with
        each error isolated (revision 3, Part 7 check). Returns a list of
        (key, status, value): status "ok" (value = fingerprint), "error" (value = message),
        or "no_base" (the query fails on D and is never priced by the sampler)."""
        tbl = self.targets.get(b)
        cur = self._new_cursor()
        name = f"{quote_ident(tbl[0])}.{quote_ident(tbl[1])}"
        contiguous, _ = self._table_facts(tbl)
        try:
            if self.neighbours == "qirana":
                self.prepare_updates([b])
                cols, eligible, dom = self._upd[b]
                if spec[0] == "row":
                    _, r, chosen, donors = spec
                    rid = self._rid(cur, name, contiguous, r)
                    names = [eligible[j] for j in chosen]
                    new = self._donor_values(cur, name, contiguous, dom, eligible, chosen, donors)
                    stmts = [(rid, names, new)]
                else:
                    _, r1, r2, chosen = spec
                    rid1, rid2 = self._rid(cur, name, contiguous, r1), self._rid(cur, name, contiguous, r2)
                    names = [eligible[j] for j in chosen]
                    sel = ", ".join(quote_ident(c) for c in names)
                    v1 = cur.execute(f"SELECT {sel} FROM {name} WHERE rowid = {rid1}").fetchone()
                    v2 = cur.execute(f"SELECT {sel} FROM {name} WHERE rowid = {rid2}").fetchone()
                    stmts = [(rid1, names, list(v2)), (rid2, names, list(v1))]

                def run():
                    for rid_, names_, vals_ in stmts:
                        sets_ = ", ".join(f"{quote_ident(x)} = ?" for x in names_)
                        cur.execute(f"UPDATE {name} SET {sets_} WHERE rowid = {rid_}", list(vals_))
            else:
                rid = self._rid(cur, name, contiguous, int(spec))

                def run():
                    cur.execute(f"DELETE FROM {name} WHERE rowid = {rid}")
            out = []
            self._apply_retry(cur, name, run)
            for k in self.readers.get(b, ()):
                if self.base_fp.get(k) is None:
                    out.append((k, "no_base", None))
                    continue
                own, duck = self.assets[k]
                try:
                    cur.execute(f"SET search_path={sql_str(search_path(own, self.schemas))}")
                    out.append((k, "ok", cur.execute(fp_sql(duck)).fetchone()))
                except Exception as e:
                    stopflag.check()
                    out.append((k, "error", f"{type(e).__name__}: {str(e)[:160]}"))
                    self._rollback(cur)
                    self._apply_retry(cur, name, run)
            self._rollback(cur)
            return out
        finally:
            cur.close()
            with self._cursor_lock:
                if cur in self._cursors:
                    self._cursors.remove(cur)

    # ---- one neighbour, shown in full (revision 3, P6 worked example) ------------------
    def neighbour_answers(self, b, spec, keys, base=False):
        """Apply one accepted neighbour inside a transaction, run the queries ``keys`` on
        it, and roll back. ``spec`` is the support entry (a row id for a deletion, an update
        spec for a Qirana neighbour). With ``base=True`` nothing is changed (answers on D).

        Returns {"table": (schema, name), "changes": [...], "answers": {key: DataFrame}}.
        The queries run with the sampler's search path, as when the conflict sets were built.
        """
        import pandas as pd
        from .database import norm_frame
        tbl = self.targets.get(b)
        if tbl is None:
            raise ValueError(f"table {b!r} is not a sampling target")
        cur = self._new_cursor()
        name = f"{quote_ident(tbl[0])}.{quote_ident(tbl[1])}"
        contiguous, _ = self._table_facts(tbl)
        changes, stmts = [], []
        try:
            if not base and self.neighbours == "qirana":
                self.prepare_updates([b])
                cols, eligible, dom = self._upd[b]
                if spec[0] == "row":
                    _, r, chosen, donors = spec
                    rid = self._rid(cur, name, contiguous, r)
                    names = [eligible[j] for j in chosen]
                    sel = ", ".join(quote_ident(c) for c in names)
                    old = cur.execute(f"SELECT {sel} FROM {name} WHERE rowid = {rid}").fetchone()
                    new = self._donor_values(cur, name, contiguous, dom, eligible, chosen, donors)
                    stmts = [(rid, names, new)]
                    changes = [{"rowid": rid, "column": c, "old": o, "new": v}
                               for c, o, v in zip(names, old, new) if not self._same(o, v)]
                else:
                    _, r1, r2, chosen = spec
                    rid1, rid2 = self._rid(cur, name, contiguous, r1), self._rid(cur, name, contiguous, r2)
                    names = [eligible[j] for j in chosen]
                    sel = ", ".join(quote_ident(c) for c in names)
                    v1 = cur.execute(f"SELECT {sel} FROM {name} WHERE rowid = {rid1}").fetchone()
                    v2 = cur.execute(f"SELECT {sel} FROM {name} WHERE rowid = {rid2}").fetchone()
                    stmts = [(rid1, names, list(v2)), (rid2, names, list(v1))]
                    changes = ([{"rowid": rid1, "column": c, "old": a, "new": d} for c, a, d in zip(names, v1, v2)
                                if not self._same(a, d)] +
                               [{"rowid": rid2, "column": c, "old": d, "new": a} for c, a, d in zip(names, v1, v2)
                                if not self._same(a, d)])
            elif not base:
                rid = self._rid(cur, name, contiguous, int(spec))
                row = cur.execute(f"SELECT * FROM {name} WHERE rowid = {rid}").fetchone()
                cols = [d[0] for d in cur.description]
                changes = [{"rowid": rid, "deleted_row": dict(zip(cols, row))}]
            cur.execute("BEGIN TRANSACTION")
            try:
                if not base and self.neighbours == "qirana":
                    for rid, names_, vals in stmts:
                        sets = ", ".join(f"{quote_ident(c)} = ?" for c in names_)
                        cur.execute(f"UPDATE {name} SET {sets} WHERE rowid = {rid}", list(vals))
                elif not base:
                    cur.execute(f"DELETE FROM {name} WHERE rowid = {changes[0]['rowid']}")
                answers = {}
                for k in keys:
                    own, duck = self.assets[k]
                    cur.execute(f"SET search_path={sql_str(search_path(own, self.schemas))}")
                    answers[k] = norm_frame(cur.execute(duck).df())
            finally:
                self._rollback(cur)
        finally:
            cur.close()
            with self._cursor_lock:
                if cur in self._cursors:
                    self._cursors.remove(cur)
        return {"table": tbl, "changes": changes, "answers": answers}

    # ---- a support set -------------------------------------------------------------
    def _result(self, fut, futs, clock, label):
        """Wait for one candidate. The stall clock restarts whenever any candidate of the
        batch finishes, so slow but moving work is never cut off."""
        poll = min(30.0, max(0.5, self.stall_s / 4))
        while True:
            try:
                return fut.result(timeout=poll)
            except FuturesTimeout:
                n_done = sum(g.done() for g in futs)
                now = time.perf_counter()
                if n_done > clock[0]:
                    clock[0], clock[1] = n_done, now
                elif now - clock[1] > self.stall_s:
                    raise SamplingStalled(
                        f"{label}: no deletion finished for {self.stall_s / 60:.0f} minutes") from None

    def sample(self, tables: list, weights, n: int, table_seed: int, row_seed: int,
               checkpoints=(), track=None, record_fp=False, max_draw_factor=20,
               label="support") -> SampleResult:
        """Accept the first ``n`` candidates, in draw order, whose deletion succeeds.

        With ``threads == 1`` candidates are evaluated in this thread, one at a time;
        otherwise a thread pool evaluates each batch ahead of time. A (table, row) drawn
        twice in one batch is evaluated once. The accepted set, and every count reported,
        is the same for any number of threads. If no candidate finishes for ``stall_s``
        seconds, SamplingStalled is raised.
        """
        accepted, incidence, changed_fp, trace, errors = [], {}, {}, [], {}
        drawn = evaluated = skipped = 0
        evaluate = self._evaluate_traced if self.trace_costs else self._evaluate
        cps = sorted(set(int(c) for c in checkpoints if c <= n))
        cp_sec = {}
        t0 = time.perf_counter()
        w = np.asarray(weights, float)
        if len(tables) == 0 or n <= 0 or not np.isfinite(w).all() or w.sum() <= 0:
            LOG.warning("%s: nothing to sample from", label)
            return SampleResult(n=0, accepted=[], incidence={}, changed_fp={}, draws=0, skipped=0,
                                seconds=0.0, checkpoint_seconds={})
        w = w / w.sum()
        if self.neighbours == "qirana":
            self.prepare_updates(tables)
        rng_t = np.random.default_rng(table_seed)
        rng_r = np.random.default_rng(row_seed)
        track = set(track) if track is not None else None
        last_log = t0
        ex = ThreadPoolExecutor(max_workers=self.threads) if self.threads > 1 else None
        abandon = False
        try:
            while len(accepted) < n and drawn < n * max_draw_factor:
                rate = len(accepted) / evaluated if evaluated else 0.7
                need = n - len(accepted)
                size = int(min(self.threads * 16, max(self.threads, math.ceil(need / max(rate, 0.05) * 1.1))))
                size = min(size, n * max_draw_factor - drawn)
                batch = []
                for _ in range(size):
                    b = tables[int(rng_t.choice(len(tables), p=w))]
                    if self.neighbours == "qirana":
                        batch.append((b, self.draw_update(b, rng_r), track, record_fp))
                        continue
                    cnt = self.counts.get(b, 0)
                    r = int(rng_r.integers(cnt)) if cnt > 0 else -1
                    batch.append((b, r, track, record_fp))
                drawn += len(batch)
                futmap, cache, futs, clock = {}, {}, [], [0, time.perf_counter()]
                if ex is not None:
                    for cand in batch:
                        if (cand[0], cand[1]) not in futmap:
                            futmap[(cand[0], cand[1])] = ex.submit(evaluate, cand)
                    futs = list(futmap.values())
                try:
                    for cand in batch:
                        if len(accepted) >= n:
                            break
                        key = (cand[0], cand[1])
                        if ex is not None:
                            res = self._result(futmap[key], futs, clock, label)
                        else:
                            if key not in cache:
                                cache[key] = evaluate(cand)
                            res = cache[key]
                        if self.trace_costs:
                            res, secs, nq = res
                        evaluated += 1
                        if res is None:
                            skipped += 1
                            if self.trace_costs:
                                trace.append((-1, round(secs, 6), int(nq)))
                            continue
                        idx = len(accepted)
                        accepted.append(key)
                        ne = self._errs.get(key, 0)     # get, not pop: a key can be drawn twice
                        if ne:
                            errors[idx] = int(ne)
                        if self.trace_costs:
                            trace.append((idx, round(secs, 6), int(nq)))
                        for k, v in res.items():
                            incidence.setdefault(k, []).append(idx)
                            if record_fp:
                                changed_fp.setdefault(k, {})[idx] = v
                        if cps and idx + 1 in cps:
                            cp_sec[idx + 1] = round(time.perf_counter() - t0, 1)
                finally:
                    for f in futs:               # candidates past the n-th are not needed
                        f.cancel()
                running = [f for f in futs if not f.done()]
                if running:                      # let started ones roll back, but never wait forever
                    _, not_done = wait(running, timeout=self.stall_s)
                    if not_done:
                        raise SamplingStalled(f"{label}: {len(not_done)} deletions did not finish "
                                              f"within {self.stall_s / 60:.0f} minutes")
                now = time.perf_counter()
                if now - last_log > 30:
                    last_log = now
                    LOG.info("%s: %d/%d neighbours accepted (%d candidates, %.0f s)", label,
                             len(accepted), n, evaluated, now - t0)
        except BaseException:
            abandon = True                       # never wait on a thread that may be stuck
            if ex is not None:
                ex.shutdown(wait=False, cancel_futures=True)
            raise
        finally:
            if not abandon:
                if ex is not None:
                    ex.shutdown(wait=True, cancel_futures=True)
                self.close()
        if len(accepted) < n:
            LOG.warning("%s: only %d of %d neighbours after %d candidates", label, len(accepted), n, evaluated)
        return SampleResult(n=len(accepted), accepted=accepted, incidence=incidence,
                            changed_fp=changed_fp, draws=evaluated, skipped=skipped,
                            seconds=round(time.perf_counter() - t0, 1), checkpoint_seconds=cp_sec,
                            cost_trace=trace, errors=errors)


def base_fingerprints(con, schemas, assets: dict) -> dict:
    out = {}
    for k, (own, duck) in assets.items():
        try:
            con.execute(f"SET search_path={sql_str(search_path(own, schemas))}")
            out[k] = con.execute(fp_sql(duck)).fetchone()
        except Exception:
            stopflag.check()
            out[k] = None
    con.execute("SET search_path='main'")
    return out


def table_counts(con, keys, targets: dict | None = None) -> dict:
    """Row counts per table key; keys that fail to count are left out (historical)."""
    out = {}
    for b in keys:
        try:
            out[b] = int(con.execute(f"SELECT count(*) FROM {quote_ident(b[0])}.{quote_ident(b[1])}").fetchone()[0])
        except Exception:
            stopflag.check()
    return out
