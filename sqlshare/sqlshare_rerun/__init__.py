"""Reproducible rerun of every SQLShare result in the paper
"Arbitrage-Free Query Pricing under Recomposition", under the setting of the main results and with every evaluation alternative switched
on (the two ends of the paper's sensitivity analysis)."""

__version__ = "3.0.0.dev11"

#: Version of the result-producing code. It keys every stage marker, so outputs made by
#: other code are never reused. Change it whenever a change can alter any result; 2.1.0
#: only added the ablation command and hooks that are off in the paper and corrected runs,
#: so results made by 2.0.0 stay valid.
RESULTS_VERSION = "2.0.0"
