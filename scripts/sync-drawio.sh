#!/usr/bin/env bash
# sync-drawio.sh — pull latest, then commit+push any .drawio edits.
# Run from anywhere inside the blog repo: `./scripts/sync-drawio.sh`
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "→ pulling latest from origin/main"
git pull --rebase --autostash

if git status -s | grep -qE '\.drawio$'; then
  echo "→ found .drawio changes, committing"
  git add '*.drawio'
  msg="${1:-edit drawio diagrams via desktop}"
  git commit -m "$msg"
  echo "→ pushing"
  git push origin main
  echo "✓ pushed — GitHub Actions will rebuild Pages in ~1 min"
else
  echo "✓ no .drawio changes"
fi
