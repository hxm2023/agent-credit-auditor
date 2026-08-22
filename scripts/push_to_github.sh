#!/usr/bin/env bash
# Publish the repo to GitHub (user-executed or authorized).
#
# Prerequisites: a GitHub repository URL (create an EMPTY repo first, e.g.
# https://github.com/<you>/agent-credit-auditor). Then either:
#   bash scripts/push_to_github.sh https://github.com/<you>/agent-credit-auditor.git
# or run the commands below manually.
set -euo pipefail
cd "$(dirname "$0")/.."
URL="${1:?usage: bash scripts/push_to_github.sh <repo-url>}"

git remote remove origin 2>/dev/null || true
git remote add origin "$URL"
git branch -M main
git push -u origin main

echo "pushed. Open the repo page and confirm:"
echo "  - CI workflow runs green (Actions tab)"
echo "  - README renders; LICENSE + CITATION.cff visible"
