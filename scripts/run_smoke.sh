#!/usr/bin/env bash
# Fast smoke test set (<1 min): everything except the full-driver pipeline
# tests. The release gate remains the FULL suite (uv run pytest).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run pytest -m "not full" -q
