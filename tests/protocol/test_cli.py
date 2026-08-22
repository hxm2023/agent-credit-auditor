"""CLI tests (design §15.1): guard validation, audit exit codes, report,
protocol validation over all five frozen protocols."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from credit_auditor.cli import main
from credit_auditor.experiments import m0 as m0_exp

ROOT = Path(__file__).resolve().parents[2]


def test_validate_protocol_all_frozen(capsys):
    protocols = sorted((ROOT / "configs/protocols").glob("*.json"))
    assert len(protocols) == 7
    for p in protocols:
        assert main(["validate-protocol", str(p)]) == 0
    out = capsys.readouterr().out
    assert out.count("mode=docs_only_semantic") == 7


def test_guard_cli_allow(tmp_path):
    env = tmp_path / "env.json"
    env.write_text(json.dumps({"schema_version": "grpo-guard-envelope-1.0", "required_extensions": [], "content_sha256": "a" * 64}), encoding="utf-8")
    assert main(["validate-guard-envelope", "--envelope", str(env)]) == 0


def test_guard_cli_reject_unknown_major(tmp_path, capsys):
    env = tmp_path / "env_bad.json"
    env.write_text(json.dumps({"schema_version": "grpo-guard-envelope-2.0", "required_extensions": [], "content_sha256": "a" * 64}), encoding="utf-8")
    assert main(["validate-guard-envelope", "--envelope", str(env)]) == 1
    assert "REJECT" in capsys.readouterr().out


def test_audit_cli_exit_codes(tmp_path):
    m0_exp.register()
    from credit_auditor import runner
    out = runner.run(
        protocol_path=ROOT / "configs/protocols/m0_regression_v1.json",
        output_dir=tmp_path / "M0",
        seed_manifests=[ROOT / "configs/seeds/m0_problems.json"],
    )
    assert main(["audit", "--artifact-dir", str(out)]) == 0
    # corrupt the package: remove a required file -> nonzero exit
    (out / "raw_rows.jsonl.zst").unlink()
    assert main(["audit", "--artifact-dir", str(out)]) == 1


def test_report_cli(tmp_path):
    m0_exp.register()
    from credit_auditor import runner
    runner.run(
        protocol_path=ROOT / "configs/protocols/m0_regression_v1.json",
        output_dir=tmp_path / "artifacts" / "M0",
        seed_manifests=[ROOT / "configs/seeds/m0_problems.json"],
    )
    # report subcommand builds into the artifact root (its pytest step is
    # skipped because build_release_report's _run_tests is excluded here via
    # the marker-less path? no — it runs tests; use the direct builder flag
    # by passing run_tests=False through a tiny wrapper is not exposed, so
    # monkeypatch the subprocess instead)
    import credit_auditor.report as report_mod
    report_mod._run_tests = lambda: "TEST_LOG skipped (test)\n"
    assert main(["report", "--artifact-root", str(tmp_path / "artifacts")]) == 0
    assert (tmp_path / "artifacts" / "REPORT.md").is_file()
