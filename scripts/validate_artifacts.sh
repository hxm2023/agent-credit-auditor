#!/usr/bin/env bash
# CI release gate: every checked-in Auditor artifact pack must pass the
# Auditor's own evidence validation (strict provenance: non-empty SHA256SUMS,
# expected-file coverage, per-file hash match, manifest fields) plus an
# OS-level sha256sum -c. P0-1 (round-1 external review): the release itself
# must survive the Auditor before it ships.
#
# Excluded: artifacts/real_scenario_demo (intentional fault-injection demo —
# its pkg is deliberately in integrity=fail state), artifacts/local (scratch).
set -euo pipefail
cd "$(dirname "$0")/.."

failed=0
checked=0
for pack in artifacts/*/*/; do
  # an Auditor pack is a directory holding a protocol plus a run manifest
  [ -f "$pack/protocol.json" ] && [ -f "$pack/run_manifest.json" ] || continue
  case "$pack" in
    artifacts/local/* | artifacts/real_scenario_demo/*) continue ;;
  esac
  checked=$((checked + 1))
  echo "== $pack"
  if ! uv run credit-auditor audit --artifact-dir "$pack"; then
    echo "FAIL(integrity): $pack"
    failed=$((failed + 1))
    continue
  fi
  if [ -f "$pack/SHA256SUMS" ]; then
    if ! (cd "$pack" && sha256sum --quiet -c SHA256SUMS); then
      echo "FAIL(sha256sum -c): $pack"
      failed=$((failed + 1))
    fi
  fi
done

if [ "$checked" -eq 0 ]; then
  echo "error: no Auditor packs found under artifacts/*/*/" >&2
  exit 1
fi
if [ "$failed" -gt 0 ]; then
  echo "error: $failed/$checked packs failed Auditor evidence validation" >&2
  exit 1
fi
echo "OK: $checked packs pass Auditor evidence validation"
