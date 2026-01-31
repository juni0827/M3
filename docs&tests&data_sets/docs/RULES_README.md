# Rule Engine Registry — Minimal DSL

A rule is an object with optional `precond` clauses composed from percentile/z-based tests.
Supported atoms:

- `{"lt_p": ["metric_name", 0.05]}`  # metric is below its 5th percentile
- `{"gt_p": ["metric_name", 0.95]}`  # metric is above its 95th percentile
- `{"z_lt": ["metric_name", -2.0]}`  # robust z-score below -2
- `{"z_gt": ["metric_name", 2.0]}`   # robust z-score above 2

`_RULES.select(context)` returns all rules whose preconditions pass using HistoryStore stats.
