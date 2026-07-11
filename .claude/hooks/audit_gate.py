#!/usr/bin/env python3
"""PreToolUse audit gate -- mechanical enforcement of mission §8 writer!=verifier.

Blocks (exit code 2) a `git commit` attempt when:
1. Staged changes touch code (src/, validation/, tests/) and no `reviewer` verdict
   has been logged to research/audit_log.jsonl since those files were last modified.
2. Staged changes touch a real trial result (research/experiments.jsonl or a
   research/**/results.json) and no `leak-hunter` verdict has been logged since.

Why this exists: invoking leak-hunter/reviewer/design-reviewer was previously
convention-only ("you must invoke X before Y") with nothing mechanical behind it --
and that already failed twice in practice (D15's referee-threshold change and D22's
unilateral flaky-test fix both bypassed subagent review), which is exactly the
"guardrails that rely on good intentions are weak" failure mode mission §3.3 warns
against. This hook does not require leak-hunter/reviewer to gain write access --
the main session logs their verdict to research/audit_log.jsonl itself, the same way
it already logs to research/journal.md and research/decisions.md.

This hook does NOT verify the logged verdict is honest (that would need write
access from the subagent itself, which conflicts with their strict read-only
design) -- it only verifies the audit step was not skipped entirely. That is the
actual failure mode observed in D22; a deliberately falsified log entry is a
different, more adversarial threat this repo's single-operator trust model does
not currently defend against.

design-reviewer's "before presenting to the operator" trigger has no analogous
tool-call to intercept (presenting is plain assistant text, not a tool call) --
that consult remains convention-only. Log it to audit_log.jsonl anyway for the
historical record, but this hook does not and cannot gate on it.

If you believe this is a false positive, stop and ask the operator -- do not
work around it (e.g. by hand-writing an audit_log.jsonl line without actually
having run the audit).
"""
import json
import os
import re
import subprocess
import sys

REPO = "/workspace/algo-trading-bot"
AUDIT_LOG = os.path.join(REPO, "research", "audit_log.jsonl")

CODE_DIRS = ("src/", "validation/", "tests/")


def deny(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(2)


def is_git_commit(command: str) -> bool:
    return bool(re.search(r"\bgit\s+(-C\s+\S+\s+)?commit\b", command))


def staged_files() -> list:
    try:
        out = subprocess.run(
            ["git", "-C", REPO, "diff", "--cached", "--name-only"],
            capture_output=True, text=True, timeout=10,
        )
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def newest_mtime(paths: list) -> float:
    best = 0.0
    for p in paths:
        try:
            best = max(best, os.path.getmtime(os.path.join(REPO, p)))
        except OSError:
            continue
    return best


def last_verdict_time(agent: str) -> float:
    if not os.path.exists(AUDIT_LOG):
        return -1.0
    best = -1.0
    try:
        with open(AUDIT_LOG) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("agent") == agent:
                    ts = row.get("unix_ts")
                    if isinstance(ts, (int, float)):
                        best = max(best, float(ts))
    except OSError:
        pass
    return best


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)
    if data.get("tool_name") != "Bash":
        sys.exit(0)
    command = str((data.get("tool_input") or {}).get("command") or "")
    if not is_git_commit(command):
        sys.exit(0)

    files = staged_files()
    if not files:
        sys.exit(0)

    code_files = [f for f in files if f.endswith(".py") and any(f.startswith(d) for d in CODE_DIRS)]
    result_files = [f for f in files if f == "research/experiments.jsonl" or f.endswith("results.json")]

    if code_files and last_verdict_time("reviewer") < newest_mtime(code_files):
        deny(
            "AUDIT GATE (mission §8 writer!=verifier): staged changes touch "
            f"{code_files} but no `reviewer` verdict in research/audit_log.jsonl "
            "postdates them. Invoke the reviewer subagent, log its verdict to "
            "research/audit_log.jsonl (unix_ts, agent, verdict, files), then retry "
            "the commit. Ask the operator if you believe this is a false positive --"
            " do not hand-write a log line without having actually run the audit."
        )

    if result_files and last_verdict_time("leak-hunter") < newest_mtime(result_files):
        deny(
            "AUDIT GATE (mission §3.3e adversarial self-audit): staged changes "
            f"touch a result file ({result_files}) but no `leak-hunter` verdict in "
            "research/audit_log.jsonl postdates it. Invoke leak-hunter, log its "
            "verdict to research/audit_log.jsonl, then retry the commit."
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
