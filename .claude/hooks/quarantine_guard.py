#!/usr/bin/env python3
"""PreToolUse quarantine guard — mechanical enforcement of mission §3.1, §3.3h, §8.

Blocks (exit code 2):
1. Any tool access whose input references the quarantined holdout store
   (``data/lockbox`` or the out-of-repo key material ``.gordian_lockbox``) —
   every attempt is appended to research/lockbox_access.log.
2. Edits to EXISTING files under specs/ (pre-registrations are frozen; new spec
   files are allowed).
3. Edits under validation/ once the referee is frozen (validation/.frozen exists).
4. Edits to the guard itself (.claude/hooks/*, .claude/settings.json) — operator only.

This hook is integrity infrastructure: Claude must never modify it (rule 4 makes
that mechanical). Operator edits happen outside Claude sessions.
"""
import datetime
import json
import os
import sys

REPO = "/workspace/algo-trading-bot"
FORBIDDEN_TOKENS = ("data/lockbox", ".gordian_lockbox", "OPERATOR_TOKEN")
GUARD_PATHS = (".claude/hooks", ".claude/settings.json")
BASH_WRITE_HINTS = (">", ">>", "sed -i", "tee ", "mv ", "cp ", "rm ", "truncate", "shred")


def deny(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(2)


def log_lockbox(line: str) -> None:
    try:
        with open(os.path.join(REPO, "research", "lockbox_access.log"), "a") as f:
            f.write(f"{datetime.datetime.now().isoformat()} {line}\n")
    except OSError:
        pass


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    tool = data.get("tool_name", "")
    ti = data.get("tool_input") or {}
    paths = [str(ti.get(k) or "") for k in ("file_path", "path", "notebook_path")]
    command = str(ti.get("command") or "")
    blob = " ".join(paths + [command])

    # --- 1. Lockbox quarantine (any tool) ---
    if any(tok in blob for tok in FORBIDDEN_TOKENS):
        allowed = tool == "Bash" and "validation/final_eval.py" in command
        if allowed:
            log_lockbox(f"ALLOWED final_eval invocation: {command[:300]!r}")
            sys.exit(0)
        log_lockbox(f"BLOCKED {tool}: {blob[:300]!r}")
        deny(
            "QUARANTINE (mission §3.1): data/lockbox holds the held-out test set and "
            "its key material. Access is forbidden until the one-shot final eval "
            "(validation/final_eval.py, operator-supplied token). Do not retry or "
            "work around this — if you believe this block is a false positive, "
            "stop and ask the operator."
        )

    editing = tool in ("Edit", "Write", "NotebookEdit")
    frozen = os.path.exists(os.path.join(REPO, "validation", ".frozen"))

    # --- 2/3/4. Write protection on frozen specs, frozen referee, and the guard ---
    if editing:
        for p in paths:
            if not p:
                continue
            rel = os.path.relpath(os.path.abspath(p), REPO)
            if rel.startswith("specs/") and os.path.exists(os.path.abspath(p)):
                deny(
                    f"FROZEN (mission §3.3c): {rel} is a pre-registration. Edits are "
                    "forbidden. Write a NEW spec file and obtain logged operator approval."
                )
            if rel.startswith("validation/") and frozen:
                deny(
                    "FROZEN (mission §8, don't shave the ruler): validation/ is the "
                    "referee and validation/.frozen exists. Changes require the "
                    "OPERATOR to remove the freeze marker themselves after a logged "
                    "approval. Do not remove it yourself."
                )
            if any(rel.startswith(g) or rel == g for g in GUARD_PATHS):
                deny(
                    f"GUARD: {rel} is integrity infrastructure. Claude may not modify "
                    "it; operator edits only."
                )

    # --- Bash sneak-path for the same write protections ---
    if tool == "Bash" and command:
        touches_specs = "specs/" in command
        touches_guard = any(g in command for g in GUARD_PATHS)
        touches_frozen_val = "validation/" in command and frozen
        looks_like_write = any(w in command for w in BASH_WRITE_HINTS)
        if looks_like_write and (touches_specs or touches_guard or touches_frozen_val):
            deny(
                "GUARD: bash writes touching specs/, .claude/ guard files, or frozen "
                "validation/ are forbidden. Use a new spec file or obtain logged "
                "operator approval. (If this is a false positive — e.g. a read-only "
                "command that merely mentions these paths — rephrase the command to "
                "avoid write-like tokens, or ask the operator.)"
            )
    sys.exit(0)


if __name__ == "__main__":
    main()
