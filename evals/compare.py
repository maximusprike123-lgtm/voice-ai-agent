"""Compare two sweeps: pass rates, counts of specific failure kinds, guard blocks per rule.

    .venv/bin/python -m evals.compare data/evals/<reference> data/evals/<new>

Reads the results.json each sweep saved. Runs are compared as counts AND per 100 runs, because
the two sweeps need not have the same number of runs. Run-to-run noise is large (the same
agent leaked a phone number in 5/50 runs in one sweep and 0/50 in the next), so read counts of
specific failure kinds, not just the headline pass rate.

Guard blocks: a sweep of an agent WITHOUT the speech guard has no "blocked" records, so its
transcripts are replayed through the guard offline ("would have been blocked"); a sweep of an
agent WITH the guard has the real ones. Either way the number is the model's raw attempts.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from agent.text_guard import SpeechGuard

SCRIPTED_TOOLS = ("prepare_booking", "confirm_booking", "take_message")  # tools that speak by code


def load(directory: Path) -> list[dict]:
    return json.loads((directory / "results.json").read_text(encoding="utf-8"))


def replay_guard(runs: list[dict]) -> Counter:
    """rule -> sentences the guard would block among the MODEL-written sentences of `runs`
    (greeting and code-built sentences after a saving/drafting tool are skipped)."""
    counts: Counter = Counter()
    for run in runs:
        guard = SpeechGuard()
        items = run["items"]
        first_scripted: dict[int, int] = {}
        for index, item in enumerate(items):
            if (
                item.get("kind") == "tool"
                and item.get("tool") in SCRIPTED_TOOLS
                and not item.get("is_error")
            ):
                first_scripted.setdefault(item.get("turn", 0), index)
        for index, item in enumerate(items):
            if item.get("kind") == "tool" and item.get("committed"):
                guard.note_commit()
            if item.get("kind") != "say" or item.get("turn", 0) == 0:
                continue
            turn = item.get("turn", 0)
            if turn in first_scripted and index > first_scripted[turn]:
                continue
            if violation := guard.check(item["text"]):
                counts[violation.rule] += 1
    return counts


def guard_blocks(runs: list[dict]) -> tuple[Counter, str]:
    """(counts, how they were obtained: "actual" or "replayed")."""
    actual: Counter = Counter()
    seen_guard = False
    for run in runs:
        for item in run["items"]:
            if item.get("kind") == "blocked":
                actual[item.get("rule", "?")] += 1
                seen_guard = True
    if seen_guard:
        return actual, "actual"
    return replay_guard(runs), "replayed"


def failure_kinds(runs: list[dict]) -> Counter:
    """check name -> number of gradable runs in which it failed."""
    counts: Counter = Counter()
    for run in runs:
        if run["status"] in ("pass", "fail"):
            for check in run["checks"]:
                if not check["passed"]:
                    counts[check["name"]] += 1
    return counts


def warning_counts(runs: list[dict]) -> Counter:
    counts: Counter = Counter()
    for run in runs:
        if run["status"] in ("pass", "fail"):
            for warning in run.get("warnings", []):
                if warning.get("fired"):
                    counts[warning["name"]] += 1
    return counts


def summarize(runs: list[dict]) -> dict:
    statuses = Counter(run["status"] for run in runs)
    per_scenario: dict[str, list[int]] = {}
    for run in runs:
        if run["status"] in ("pass", "fail"):
            cell = per_scenario.setdefault(run["scenario"], [0, 0])
            cell[1] += 1
            cell[0] += run["status"] == "pass"
    return {
        "runs": len(runs),
        "gradable": statuses["pass"] + statuses["fail"],
        "passed": statuses["pass"],
        "infra": statuses["infra_error"],
        "inconclusive": statuses["inconclusive"],
        "per_scenario": per_scenario,
        "failures": failure_kinds(runs),
        "warnings": warning_counts(runs),
        "guard": guard_blocks(runs),
    }


def _rate(passed: int, total: int) -> str:
    return f"{passed}/{total} = {passed / total:.0%}" if total else "n/a"


def _cell(count: int, runs: int) -> str:
    return f"{count:>3} ({100 * count / runs:>4.0f}/100)" if runs else "  -"


def format_comparison(ref: dict, new: dict, ref_name: str, new_name: str) -> str:
    lines = [
        "",
        f"{'':<38}{ref_name:>22}{new_name:>22}",
        f"{'runs (gradable, infra, inconclusive)':<38}"
        f"{ref['runs']:>10} ({ref['gradable']},{ref['infra']},{ref['inconclusive']})".rjust(22)
        + f"{new['runs']:>10} ({new['gradable']},{new['infra']},{new['inconclusive']})".rjust(22),
        f"{'raw pass rate':<38}"
        f"{_rate(ref['passed'], ref['gradable']):>22}{_rate(new['passed'], new['gradable']):>22}",
        "",
        "PASS RATE PER SCENARIO",
    ]
    for sid in sorted(set(ref["per_scenario"]) | set(new["per_scenario"])):
        r = ref["per_scenario"].get(sid, [0, 0])
        n = new["per_scenario"].get(sid, [0, 0])
        lines.append(f"  {sid:<36}{_rate(*r):>22}{_rate(*n):>22}")

    def block(title: str, ref_counts: Counter, new_counts: Counter, note: str = "") -> None:
        lines.extend(["", title + (f"  {note}" if note else "")])
        names = sorted(
            set(ref_counts) | set(new_counts), key=lambda n: -(ref_counts[n] + new_counts[n])
        )
        if not names:
            lines.append("  (none)")
        for name in names:
            lines.append(
                f"  {name:<36}{_cell(ref_counts[name], ref['gradable']):>22}"
                f"{_cell(new_counts[name], new['gradable']):>22}"
            )

    block(
        "FAILED CHECKS (runs in which the check failed; count and per 100 gradable runs)",
        ref["failures"],
        new["failures"],
    )
    ref_blocks, ref_how = ref["guard"]
    new_blocks, new_how = new["guard"]
    block(
        "SPEECH GUARD BLOCKS (sentences; count and per 100 gradable runs)",
        ref_blocks,
        new_blocks,
        f"[{ref_name}: {ref_how}, {new_name}: {new_how}]",
    )
    block("WARNING METRICS (runs in which the warning fired)", ref["warnings"], new["warnings"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.compare", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument("reference", type=Path, help="results directory of the reference sweep")
    parser.add_argument("new", type=Path, help="results directory of the sweep to compare")
    args = parser.parse_args(argv)
    try:
        ref, new = summarize(load(args.reference)), summarize(load(args.new))
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"cannot read the results: {exc}", file=sys.stderr)
        return 1
    print(format_comparison(ref, new, "reference", "new"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
