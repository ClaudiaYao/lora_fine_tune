#!/usr/bin/env bash
# Create the two virtual environments used for the comparison.
#   ./setup_venvs.sh          both
#   ./setup_venvs.sh hf       .venv-hf       (Transformers + PEFT)
#   ./setup_venvs.sh unsloth  .venv-unsloth  (Unsloth)
# Exact installed versions are saved to requirements-lock-<name>.txt; keep them with the results.
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${PYTHON:-python3}"

make_env() {
  local name="$1" dir=".venv-$1"
  echo "=== $dir ==="
  "$PYTHON" -m venv "$dir"
  "$dir/bin/python" -m pip install --upgrade pip
  "$dir/bin/python" -m pip install -r "requirements-$name.txt"
  "$dir/bin/python" -m pip freeze > "requirements-lock-$name.txt"
  echo "installed; exact versions saved to requirements-lock-$name.txt"
}

case "${1:-all}" in
  hf)      make_env hf ;;
  unsloth) make_env unsloth ;;
  all)     make_env hf; make_env unsloth ;;
  *)       echo "usage: $0 [hf|unsloth|all]" >&2; exit 1 ;;
esac
