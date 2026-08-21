"""Independent oracles (design §10).

Oracle scripts are SELF-CONTAINED: they import nothing from `credit_auditor`,
read one JSON world spec on stdin, write one JSON result on stdout. They must
run in an independent Python process (§10.1). Isolation is verified by
`oracles.isolation` (AST import-graph check) and by import-monkeypatch tests.
"""
