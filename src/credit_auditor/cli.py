"""CLI (design §15.1)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from credit_auditor import runner
from credit_auditor.experiments import m0 as _m0

_m0.register()


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

    p_report = sub.add_parser("report", help="build release report from artifact root")
    p_report.add_argument("--artifact-root", required=True, type=Path)
    p_report.add_argument("--output", type=Path, default=None)

    args = parser.parse_args(argv)

    if args.command == "validate-protocol":
        proto = runner.validate_protocol(args.protocol)
        print(f"OK {args.protocol} -> protocol_id={proto.protocol_id} v{proto.protocol_version} "
              f"mode={proto.reconstruction_mode}")
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
        print(f"integrity={decision.get('integrity', 'unknown')}")
        return 0

    if args.command == "report":
        from credit_auditor.report import build_release_report
        out = build_release_report(args.artifact_root, args.output)
        print(f"report written: {out}")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
