"""Oracle import-graph isolation checks (§10.1, §17.2).

The oracle modules must not import `credit_auditor.estimators` (or anything
under `credit_auditor`) — oracle targets and estimator distributions must be
produced by different modules with no shared code.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

FORBIDDEN_TOP_LEVEL = {"credit_auditor"}
# Oracles are allowed zero project imports; they are self-contained programs.


def imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def check_import_isolation(module_path: Path) -> list[str]:
    """Return forbidden imports found in an oracle module ([] == isolated)."""
    bad: list[str] = []
    for mod in imports_of(module_path):
        top = mod.split(".")[0]
        if top in FORBIDDEN_TOP_LEVEL:
            bad.append(mod)
    return bad


def check_process_import_isolation(module_path: Path, python: str | None = None) -> dict:
    """Run the oracle module in a subprocess, print its sys.modules top-levels,
    and assert no `credit_auditor` anywhere in its import graph."""
    import json
    import subprocess

    probe = (
        "import json, subprocess, sys, importlib.util;"
        f"spec=importlib.util.spec_from_file_location('_oracle_probe', {str(module_path)!r});"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);"
        "tops=sorted({k.split('.')[0] for k in sys.modules});"
        "print(json.dumps({'tops': tops, 'has_credit_auditor': 'credit_auditor' in tops}))"
    )
    proc = subprocess.run([python or sys.executable, "-c", probe], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        return {"isolated": False, "error": proc.stderr[-2000:]}
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    isolated = not out["has_credit_auditor"]
    return {"isolated": isolated, "tops": out["tops"]}
