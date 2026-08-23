"""CLI (design §15.1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from credit_auditor import runner
from credit_auditor.experiments import continuation as _continuation
from credit_auditor.experiments import continuation_scale as _continuation_scale
from credit_auditor.experiments import d002 as _d002
from credit_auditor.experiments import m0 as _m0
from credit_auditor.experiments import minimal_logging as _minimal_logging
from credit_auditor.experiments import self_audit as _self_audit
from credit_auditor.experiments import v001 as _v001

_m0.register()
_v001.register()
_d002.register()
_continuation.register()
_continuation_scale.register()
_minimal_logging.register()
_self_audit.register()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="credit-auditor", description="Agent-RL Credit Auditor")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate-protocol", help="validate a frozen protocol file")
    p_validate.add_argument("protocol", type=Path)

    p_run = sub.add_parser("run", help="run a protocol phase")
    p_run.add_argument("--protocol", required=True, type=Path)
    p_run.add_argument("--output", required=True, type=Path)
    p_run.add_argument("--phase", default="run", choices=["run", "calibration", "test"])
    p_run.add_argument("--frozen-selection", type=Path, default=None)
    p_run.add_argument("--seed", action="append", type=Path, default=[], help="seed manifest (repeatable)")

    p_audit = sub.add_parser("audit", help="audit an artifact directory")
    p_audit.add_argument("--artifact-dir", required=True, type=Path)

    p_guard = sub.add_parser(
        "validate-guard-envelope", help="fail-closed validation of a GRPO-Guard envelope (design 25)"
    )
    p_guard.add_argument("--envelope", required=True, type=Path)

    p_legacy = sub.add_parser("validate-legacy-bundle", help="validate a legacy migration bundle (design 13.6)")
    p_legacy.add_argument("--bundle", required=True, type=Path)
    p_legacy.add_argument("--anchor", default=None, help="out-of-band root sha256 trust anchor")

    p_report = sub.add_parser("report", help="build release report from artifact root")
    p_report.add_argument("--artifact-root", required=True, type=Path)
    p_report.add_argument("--output", type=Path, default=None)
    p_report.add_argument(
        "--skip-tests", action="store_true", help="skip the embedded full test-suite run for TEST_LOG.txt (fast)"
    )

    args = parser.parse_args(argv)

    if args.command == "validate-protocol":
        proto = runner.validate_protocol(args.protocol)
        print(
            f"OK {args.protocol} -> protocol_id={proto.protocol_id} {proto.protocol_version} "
            f"mode={proto.reconstruction_mode}"
        )
        return 0

    if args.command == "run":
        out = runner.run(
            protocol_path=args.protocol,
            output_dir=args.output,
            phase=args.phase,
            frozen_selection=args.frozen_selection,
            seed_manifests=list(args.seed),
            argv=sys.argv,
        )
        print(f"published {out}")
        return 0

    if args.command == "audit":
        from credit_auditor.audit.provenance import audit_artifact_dir

        decision = audit_artifact_dir(args.artifact_dir)
        gd_path = args.artifact_dir / "gate_decision.json"
        claims: list[str] = []
        if gd_path.is_file():
            import json as _json

            gd = _json.loads(gd_path.read_text(encoding="utf-8"))
            for c in gd.get("claims", []):
                ceiling = c.get("claim_ceiling", {}).get("forbidden", [])
                suffix = f"  ceiling: {ceiling[0]}" if ceiling else ""
                claims.append(f"  {c['claim_id']}: {c['status']}{suffix}")
        print(f"integrity={decision['integrity']}")
        for line in claims:
            print(line)
        if decision["errors"]:
            print("errors:")
            for e in decision["errors"]:
                print(f"  - {e}")
        # §12: INVALID evidence means the experiment cannot support claims
        return 1 if decision["integrity"] != "pass" else 0

    if args.command == "validate-guard-envelope":
        import json as _json

        from credit_auditor.adapters.grpo_guard_envelope import validate_guard_envelope

        envelope = _json.loads(args.envelope.read_text(encoding="utf-8"))
        out = validate_guard_envelope(envelope)
        print(f"status={out['status']}" + (f" reason={out['reason']}" if "reason" in out else ""))
        return 0 if out["status"] == "ALLOW" else 1

    if args.command == "validate-legacy-bundle":
        from credit_auditor.adapters.legacy_bundle import validate_legacy_bundle

        out = validate_legacy_bundle(args.bundle, anchor_root_sha256=args.anchor)
        print(f"status={out['status']}")
        for r in out.get("reasons", out.get("structural_reasons", [])):
            print(f"  - {r}")
        if out.get("root_sha256"):
            print(f"root_sha256={out['root_sha256']}")
        return 0 if out["status"] == "VALID" else 1

    if args.command == "report":
        from credit_auditor.report import build_release_report

        out = build_release_report(args.artifact_root, args.output, run_tests=not args.skip_tests)
        print(f"report written: {out}")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
