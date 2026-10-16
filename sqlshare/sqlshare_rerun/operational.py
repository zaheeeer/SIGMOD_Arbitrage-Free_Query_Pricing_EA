"""Operational buyer (revision 3, supervisor comment 3, decision D3).

The 2.x adversary (``planner.plan_all``) chooses its sources by comparing their answers
with the target answer, so it uses information a buyer does not have. This module defines
a buyer that plans from PUBLIC information only:

    - the text of every query in the catalog (and the column list of every table it reads),
    - the posted prices,

and, after buying, the answers of the purchased sources. The buyer accepts a plan only
when the query texts GUARANTEE that the plan rebuilds the target on every database
instance (a schema-level guarantee within a restricted query fragment), not merely on the
current instance.

Fragment (``parse_query``): one base table (or an alias view of one), a select list of
plain column references (optionally renamed) or ``*``, optional DISTINCT, and a WHERE
clause that is a conjunction of comparisons of one column with constants (=, <, <=, >,
>=, IN, BETWEEN). Anything else (joins, grouping, aggregates, functions, OR, NOT, LIKE,
LIMIT/TOP, subqueries, views with their own logic) is outside the fragment, and the
operational buyer does not attack such a target.

Guarantee (``target_structure``): a source s can contribute to target t when both read the
same base table, every column t outputs is in s's output, and either s's region lies
inside t's region (then s's rows are t-rows) or every column of t's predicate is in s's
output (then the buyer filters s's answer by t's predicate, which is public). A set of
sources rebuilds a DISTINCT target when their regions cover t's region; a bag target is
rebuilt only by one bag source whose region contains t's region. Regions are compared
exactly by splitting each predicate column into atoms (points, open intervals, NULL,
"other values").

Information boundary (supervisor comment 3, "precise demonstration"):
``plan_operational`` receives prices and the structure built from query texts only; it
never receives an answer. ``execute_plan`` receives only the answers of the purchased
sources, through ``PurchasedAnswers``, which refuses any other key. The verifier compares
the result with the target answer afterwards. ``tests/test_operational.py`` checks both.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NULL = ("NULL",)        # atom: the NULL value
OTHER = ("OTHER",)      # atom: every value not named in any constraint of a set column


# ---------------------------------------------------------------------------------------
# constraints on one column
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Constraint:
    """Allowed non-NULL values of one column: an interval, intersected with a finite set
    when ``values`` is not None. NULL never satisfies a comparison."""
    lo: object = None
    lo_inc: bool = True
    hi: object = None
    hi_inc: bool = True
    values: frozenset | None = None

    def contains_value(self, v) -> bool:
        if v is None:
            return False
        try:
            if self.lo is not None and (v < self.lo or (v == self.lo and not self.lo_inc)):
                return False
            if self.hi is not None and (v > self.hi or (v == self.hi and not self.hi_inc)):
                return False
        except TypeError:
            return False
        return self.values is None or v in self.values

    def intersect(self, other: "Constraint") -> "Constraint":
        lo, lo_inc = self.lo, self.lo_inc
        if other.lo is not None and (lo is None or other.lo > lo or (other.lo == lo and not other.lo_inc)):
            lo, lo_inc = other.lo, other.lo_inc
        hi, hi_inc = self.hi, self.hi_inc
        if other.hi is not None and (hi is None or other.hi < hi or (other.hi == hi and not other.hi_inc)):
            hi, hi_inc = other.hi, other.hi_inc
        if self.values is None:
            vals = other.values
        elif other.values is None:
            vals = self.values
        else:
            vals = self.values & other.values
        c = Constraint(lo, lo_inc, hi, hi_inc, None)
        if vals is not None:
            vals = frozenset(v for v in vals if c.contains_value(v))
        return Constraint(lo, lo_inc, hi, hi_inc, vals)

    def is_empty(self) -> bool:
        if self.values is not None:
            return len(self.values) == 0
        if self.lo is not None and self.hi is not None:
            if self.lo > self.hi:
                return True
            if self.lo == self.hi and not (self.lo_inc and self.hi_inc):
                return True
        return False


# ---------------------------------------------------------------------------------------
# parsed query
# ---------------------------------------------------------------------------------------
@dataclass
class QueryForm:
    key: object
    ok: bool
    reason: str = ""
    table: tuple | None = None            # (schema, name) of the base table, lowercased
    out: tuple = ()                       # ((output name, base column lowercased), ...)
    distinct: bool = False
    box: dict = field(default_factory=dict)   # base column -> Constraint

    @property
    def base_cols(self) -> set:
        return {b for _, b in self.out}


def _const(e):
    from sqlglot import exp
    if isinstance(e, exp.Neg) and isinstance(e.this, exp.Literal) and not e.this.is_string:
        v = _const(e.this)
        return None if v is None else -v
    if isinstance(e, exp.Literal):
        if e.is_string:
            return e.this
        try:
            f = float(e.this)
        except ValueError:
            return None
        return int(f) if f.is_integer() and "." not in e.this and "e" not in e.this.lower() else f
    return None


def _colname(e, alias_ok):
    from sqlglot import exp
    if not isinstance(e, exp.Column):
        return None
    if e.table and e.table.lower() not in alias_ok:
        return None
    return e.name.lower()


def _atoms(where, alias_ok):
    """Conjunction of single-column comparisons -> {col: Constraint}, or None."""
    from sqlglot import exp
    if where is None:
        return {}
    conj = list(where.flatten()) if isinstance(where, exp.And) else [where]
    box = {}

    def add(col, c):
        box[col] = box[col].intersect(c) if col in box else c

    for a in conj:
        if isinstance(a, exp.Paren):
            a = a.this
            if isinstance(a, exp.And):
                sub = _atoms(a, alias_ok)
                if sub is None:
                    return None
                for k, v in sub.items():
                    add(k, v)
                continue
        if isinstance(a, exp.In):
            col = _colname(a.this, alias_ok)
            vals = [_const(x) for x in a.expressions]
            if col is None or not vals or any(v is None for v in vals) or a.args.get("query"):
                return None
            add(col, Constraint(values=frozenset(vals)))
            continue
        if isinstance(a, exp.Between):
            col = _colname(a.this, alias_ok)
            lo, hi = _const(a.args.get("low")), _const(a.args.get("high"))
            if col is None or lo is None or hi is None:
                return None
            add(col, Constraint(lo, True, hi, True))
            continue
        ops = {exp.EQ: "=", exp.LT: "<", exp.LTE: "<=", exp.GT: ">", exp.GTE: ">="}
        op = next((o for t, o in ops.items() if isinstance(a, t)), None)
        if op is None:
            return None
        l, r = a.this, a.expression
        col, val = _colname(l, alias_ok), _const(r)
        if col is None or val is None:
            col, val = _colname(r, alias_ok), _const(l)
            op = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "="}[op]
            if col is None or val is None:
                return None
        c = {"=": Constraint(values=frozenset([val])), "<": Constraint(hi=val, hi_inc=False),
             "<=": Constraint(hi=val, hi_inc=True), ">": Constraint(lo=val, lo_inc=False),
             ">=": Constraint(lo=val, lo_inc=True)}[op]
        add(col, c)
    for c in box.values():
        kinds = {type(v) for v in (c.values or ())} | {type(x) for x in (c.lo, c.hi) if x is not None}
        kinds = {float if k is int else k for k in kinds}
        if len(kinds) > 1 or c.is_empty():
            return None
    return box


def parse_query(key, duck_sql: str, resolve_table, table_columns, default_schema: str = "") -> QueryForm:
    """Parse one query into the fragment.

    resolve_table(schema, name) -> (schema, name) of the base table (alias views resolved),
    or None when the object is not a base table or an alias of one.
    table_columns(base) -> list of the base table's column names (the public schema).
    """
    try:
        import sqlglot
        from sqlglot import exp
        tree = sqlglot.parse_one(duck_sql, read="duckdb")
    except Exception:
        return QueryForm(key, False, "parse")
    if not isinstance(tree, exp.Select):
        return QueryForm(key, False, "not a plain SELECT")
    for arg in ("group", "having", "qualify", "with", "limit", "offset", "joins", "laterals", "windows"):
        if tree.args.get(arg):
            return QueryForm(key, False, f"has {arg}")
    froms = tree.args.get("from_") or tree.args.get("from")
    if froms is None or not isinstance(froms.this, exp.Table):
        return QueryForm(key, False, "no single table")
    t = froms.this
    base = resolve_table(t.db or default_schema, t.name)
    if base is None:
        return QueryForm(key, False, "not a base table or alias")
    base = (base[0].lower(), base[1].lower())
    alias_ok = {t.name.lower()} | ({t.alias.lower()} if t.alias else set())
    if any(isinstance(n, (exp.Subquery, exp.AggFunc, exp.Window)) for n in tree.walk()):
        return QueryForm(key, False, "subquery or aggregate")
    orig = list(table_columns(base))
    cols = [c.lower() for c in orig]
    out = []
    for e in tree.expressions:
        if isinstance(e, exp.Star) or (isinstance(e, exp.Column) and isinstance(e.this, exp.Star)):
            out += [(o, o.lower()) for o in orig]
            continue
        inner = e.this if isinstance(e, exp.Alias) else e
        c = _colname(inner, alias_ok)
        if c is None or c not in cols:
            return QueryForm(key, False, "select list is not plain columns")
        out.append((e.alias_or_name, c))     # output name as DuckDB reports it
    if len({o.lower() for o, _ in out}) != len(out):
        return QueryForm(key, False, "duplicate output names")
    box = _atoms(tree.args.get("where").this if tree.args.get("where") else None, alias_ok)
    if box is None:
        return QueryForm(key, False, "predicate outside the fragment")
    if any(c not in cols for c in box):
        return QueryForm(key, False, "unknown predicate column")
    distinct = bool(tree.args.get("distinct"))
    return QueryForm(key, True, "", base, tuple(out), distinct, box)


# ---------------------------------------------------------------------------------------
# regions and covers
# ---------------------------------------------------------------------------------------
def _column_atoms(col, constraints):
    """Atoms of one column such that every constraint is a union of atoms."""
    cs = [c for c in constraints if c is not None]
    pts = set()
    interval = False
    for c in cs:
        if c.values is not None:
            pts |= set(c.values)
        for x in (c.lo, c.hi):
            if x is not None:
                pts.add(x)
                interval = True
        if c.values is None and c.lo is None and c.hi is None:
            interval = True
    try:
        pts = sorted(pts)
    except TypeError:
        return None
    atoms = [NULL]
    if not interval:                    # only finite sets: named values and "the rest"
        return atoms + [("P", p) for p in pts] + [OTHER]
    if not pts:
        return atoms + [("I", None, None)]
    atoms.append(("I", None, pts[0]))
    for a, b in zip(pts, pts[1:]):
        atoms += [("P", a), ("I", a, b)]
    atoms += [("P", pts[-1]), ("I", pts[-1], None)]
    return atoms


def _atom_in(atom, c: Constraint | None) -> bool:
    if c is None:                       # unconstrained column: every atom, NULL included
        return True
    if atom == NULL or atom == OTHER:
        return False
    if atom[0] == "P":
        return c.contains_value(atom[1])
    _, a, b = atom                      # open interval (a, b): inside c if within bounds
    if c.values is not None:
        return False
    if c.lo is not None and (a is None or a < c.lo):
        return False
    if c.hi is not None and (b is None or b > c.hi):
        return False
    return True


def target_structure(t: QueryForm, forms: dict, max_cells: int = 20_000):
    """Sources that can contribute to target t and the region cells each one covers.

    Returns (contrib, n_cells, filters, kind) or None when t is outside the fragment:
    contrib[s] = array of cell indices of t's region that s's purchased answer supplies,
    filters[s] = True when the buyer must filter s's answer by t's predicate,
    kind = "set" (DISTINCT target, union covers allowed) or "bag" (one source only).
    """
    if not t.ok:
        return None
    tcols = t.base_cols
    cand = {}
    for s, f in forms.items():
        if s == t.key or not f.ok or f.table != t.table or not tcols <= f.base_cols:
            continue
        if t.distinct is False and f.distinct:
            continue                    # a DISTINCT source cannot rebuild a bag target
        inside = _region_inside(f.box, t.box)
        filterable = set(t.box) <= f.base_cols
        if inside or filterable:
            cand[s] = (f, not inside)
    if not cand:
        return {}, 0, {}, ("set" if t.distinct else "bag")
    cols = sorted(set(t.box) | {c for f, _ in cand.values() for c in f.box})
    per_col = []
    for c in cols:
        atoms = _column_atoms(c, [t.box.get(c)] + [f.box.get(c) for f, _ in cand.values()])
        if atoms is None:
            return None
        per_col.append([a for a in atoms if _atom_in(a, t.box.get(c))])
    n_cells = math.prod(len(a) for a in per_col) if per_col else 1
    if n_cells == 0:
        return None                     # the target's region is empty: always the empty answer
    if n_cells > max_cells:
        return None
    cells = list(itertools.product(*per_col)) if per_col else [()]
    contrib, filters = {}, {}
    for s, (f, filt) in cand.items():
        idx = [i for i, cell in enumerate(cells)
               if all(_atom_in(a, f.box.get(c)) for a, c in zip(cell, cols))]
        if idx:
            contrib[s] = np.array(idx, dtype=int)
            filters[s] = filt
    kind = "set" if t.distinct else "bag"
    if kind == "bag":                   # one source must cover the whole region
        contrib = {s: v for s, v in contrib.items() if len(v) == n_cells}
        filters = {s: filters[s] for s in contrib}
    return contrib, n_cells, filters, kind


def _region_inside(inner: dict, outer: dict) -> bool:
    """Region of ``inner`` (conjunction) inside region of ``outer``, column by column."""
    for c, oc in outer.items():
        ic = inner.get(c)
        if ic is None:
            return False                # inner allows anything (and NULL) on c
        if ic.values is not None:
            if not all(oc.contains_value(v) for v in ic.values):
                return False
            continue
        if oc.values is not None:
            return False                # an interval is never inside a finite set
        if oc.lo is not None:
            if ic.lo is None or ic.lo < oc.lo or (ic.lo == oc.lo and ic.lo_inc and not oc.lo_inc):
                return False
        if oc.hi is not None:
            if ic.hi is None or ic.hi > oc.hi or (ic.hi == oc.hi and ic.hi_inc and not oc.hi_inc):
                return False
    return True


def semantic_pairs(structures: dict) -> list:
    """(source, target) pairs where one source alone is guaranteed to rebuild the target:
    the query-text (schema-level) relation, as opposed to the instance-level one."""
    out = []
    for t, st in structures.items():
        if not st:
            continue
        contrib, n_cells, _, _ = st
        for s, idx in contrib.items():
            if n_cells and len(idx) == n_cells:
                out.append((s, t))
    return out


# ---------------------------------------------------------------------------------------
# planning (no answers) and execution (purchased answers only)
# ---------------------------------------------------------------------------------------
def plan_operational(prices: dict, structures: dict, cover_fn):
    """Cheapest guaranteed plan for every target, from prices and query structure only.

    prices: key -> posted price. structures: key -> target_structure(...) result.
    cover_fn(contrib, n, prices) -> (cost, chosen sources), e.g. planner.min_cost_row_cover.
    Returns {target: (cost, [sources])} for targets with a plan cheaper than their price.
    """
    plans = {}
    for t, st in structures.items():
        if not st:
            continue
        contrib, n_cells, _, kind = st
        pt = prices[t]
        cheap = {s: v for s, v in contrib.items() if prices[s] < pt - 1e-9}
        if not cheap or n_cells == 0:
            continue
        covered = np.zeros(n_cells, bool)
        for v in cheap.values():
            covered[v] = True
        if not covered.all():
            continue
        if kind == "bag":
            s = min(cheap, key=lambda x: (prices[x], str(x)))
            cost, plan = prices[s], [s]
        else:
            cost, plan = cover_fn(cheap, n_cells, prices)
        if plan and cost < pt - 1e-9:
            plans[t] = (float(cost), list(plan))
    return plans


class PurchasedAnswers:
    """The answers a buyer holds: only those of the purchased sources."""

    def __init__(self, answers: dict, purchased):
        self._d = {s: answers[s] for s in purchased}

    def __getitem__(self, k):
        if k not in self._d:
            raise KeyError(f"answer of {k!r} was not purchased")
        return self._d[k]

    def keys(self):
        return self._d.keys()


def answer_name(name) -> str:
    """Column name as the pipeline stores answers (``database.norm_frame``: stripped and
    lowercased). Output names from query texts are compared in this form (Part 5 fix:
    mixed-case names in the SQLShare catalog failed every plan on the real data)."""
    return str(name).strip().lower()


def _filter_sql(t: QueryForm, f: QueryForm, view: str) -> str:
    """Row ids of source f's answer that satisfy t's predicate (public query text)."""
    name_of = {b: answer_name(o) for o, b in f.out}

    def lit(v):
        if isinstance(v, str):
            return "'" + v.replace("'", "''") + "'"
        return repr(v)

    parts = []
    for c, con in t.box.items():
        q = '"' + name_of[c].replace('"', '""') + '"'
        if con.values is not None:
            parts.append(f"{q} IN ({', '.join(lit(v) for v in sorted(con.values))})")
            continue
        if con.lo is not None:
            parts.append(f"{q} {'>=' if con.lo_inc else '>'} {lit(con.lo)}")
        if con.hi is not None:
            parts.append(f"{q} {'<=' if con.hi_inc else '<'} {lit(con.hi)}")
        if con.lo is None and con.hi is None:
            parts.append(f"{q} IS NOT NULL")
    return f'SELECT "__rid__" FROM {view}' + (" WHERE " + " AND ".join(parts) if parts else "")


def execute_plan(t: QueryForm, plan, forms: dict, filters: dict, held: PurchasedAnswers):
    """Rebuild t from the purchased answers: filter where needed, project, union.

    DuckDB evaluates t's predicate (SQL semantics) but only selects row ids; the rows are
    then taken from the purchased frame itself, so values and column types are unchanged.
    """
    import duckdb
    con = duckdb.connect()
    parts = []
    out_names = [answer_name(o) for o, _ in t.out]
    try:
        for s in plan:
            f = forms[s]
            S = held[s]
            if filters.get(s):
                src = S.reset_index(drop=True)
                tagged = src.assign(__rid__=np.arange(len(src)))
                con.register("__src__", tagged)
                rid = con.execute(_filter_sql(t, f, "__src__")).fetchnumpy()["__rid__"]
                con.unregister("__src__")
                S = src.iloc[np.sort(np.asarray(rid, dtype=np.int64))]
            name_of = {b: answer_name(o) for o, b in f.out}
            P = S.loc[:, [name_of[b] for _, b in t.out]]
            P.columns = out_names
            parts.append(P)
    finally:
        con.close()
    got = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=out_names)
    if t.distinct:
        got = got.drop_duplicates()
    return got.reset_index(drop=True)


def _align_types(got: pd.DataFrame, want: pd.DataFrame) -> pd.DataFrame:
    """A union of sources can change a column's type (for example an integer column with
    no NULL in one source and NULLs in another). Where one side is numeric and the other
    holds only numbers and nulls, the other side is read as numbers, without losing any
    value; nothing else is changed."""
    got = got.copy()
    for c in got.columns:
        if c not in want.columns:
            continue
        g, w = got[c], want[c]
        if pd.api.types.is_numeric_dtype(w) and not pd.api.types.is_numeric_dtype(g):
            num = pd.to_numeric(g, errors="coerce")
            if int(num.notna().sum()) == int(g.notna().sum()):
                got[c] = num
    return got


def compare_rebuilt(got: pd.DataFrame, want: pd.DataFrame):
    """(equal, note): the verifier's check of a rebuilt target, with the reason when it
    fails (logged in plans_log.csv)."""
    from .lattice import frames_equal
    want = want.reset_index(drop=True)
    if list(got.columns) != list(want.columns):
        return False, f"columns {list(got.columns)[:6]} vs {list(want.columns)[:6]}"
    if got.shape != want.shape:
        return False, f"shape {got.shape} vs {want.shape}"
    if frames_equal(got, want):
        return True, "ok"
    g2 = _align_types(got, want)
    w2 = _align_types(want, got)
    if frames_equal(g2, w2):
        return True, "ok (numeric types aligned)"
    return False, "values differ"
