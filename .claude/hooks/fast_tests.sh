#!/usr/bin/env bash
# PostToolUse: run the fast unit-test suite after source edits (mission §9).
# No-ops silently until tests/fast exists. Exit 2 => feedback shown to Claude.
REPO=/workspace/algo-trading-bot
[ -d "$REPO/tests/fast" ] || exit 0
cd "$REPO" || exit 0
OUT=$(timeout 120 /workspace/venv/bin/python -m pytest tests/fast -q -x 2>&1)
if [ $? -ne 0 ]; then
  echo "FAST TESTS FAILED after your edit — fix before continuing (mission §8: keep the build green):" >&2
  echo "$OUT" | tail -25 >&2
  exit 2
fi
exit 0
