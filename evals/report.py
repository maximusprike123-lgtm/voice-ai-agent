"""Turning graded runs into the numbers and text a person reads."""

import json
from dataclasses import dataclass, field

from evals.checks import INVARIANT_NAMES, WARNING_NAMES
from evals.model import CheckResult, RunResult

OK_MARK, BAD_MARK = "✓", "✗"


@dataclass
class Graded:
    run: RunResult
    results: list[CheckResult]
    warnings: list[CheckResult] = field(default_factory=list)  # reported, never pass/fail

    @property
    def status(self) -> str:
        """pass / fail for gradable runs; infra_error / inconclusive are left out of the rates."""
        if self.run.outcome != "completed":
            return self.run.outcome
        return "pass" if all(r.passed for r in self.results) else "fail"

    @property
    def failed_checks(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]


@dataclass
class ScenarioSummary:
    scenario_id: str
    passed: int = 0
    failed: int = 0
    infra: int = 0
    inconclusive: int = 0
    turns: float = 0.0
    seconds: float = 0.0
    checks: dict[str, list[int]] | None = None  # name -> [passed, total] over gradable runs
    examples: dict[str, str] | None = None  # name -> one failure detail
    warnings: dict[str, list[int]] | None = None  # name -> [runs with the warning, gradable runs]
    warning_examples: dict[str, str] | None = None
    markers_ignored: int = 0

    @property
    def gradable(self) -> int:
        return self.passed + self.failed

    @property
    def runs(self) -> int:
        return self.gradable + self.infra + self.inconclusive

    @property
    def pass_rate(self) -> float | None:
        return self.passed / self.gradable if self.gradable else None


def summarize(graded: list[Graded], order: list[str]) -> dict[str, ScenarioSummary]:
    summaries = {
        sid: ScenarioSummary(sid, checks={}, examples={}, warnings={}, warning_examples={})
        for sid in order
    }
    turn_totals: dict[str, list[int]] = {sid: [] for sid in order}
    for g in graded:
        s = summaries[g.run.scenario_id]
        status = g.status
        if status == "pass":
            s.passed += 1
        elif status == "fail":
            s.failed += 1
        elif status == "infra_error":
            s.infra += 1
        else:
            s.inconclusive += 1
        s.seconds += g.run.seconds
        s.markers_ignored += g.run.markers_ignored
        turn_totals[g.run.scenario_id].append(g.run.turns)
        if status in ("pass", "fail"):
            for w in g.warnings:
                counts = s.warnings.setdefault(w.name, [0, 0])
                counts[1] += 1
                counts[0] += not w.passed
                if not w.passed:
                    s.warning_examples.setdefault(w.name, w.detail)
            for r in g.results:
                counts = s.checks.setdefault(r.name, [0, 0])
                counts[1] += 1
                counts[0] += r.passed
                if not r.passed:
                    s.examples.setdefault(r.name, r.detail)
    for sid, s in summaries.items():
        n = len(turn_totals[sid])
        s.turns = sum(turn_totals[sid]) / n if n else 0.0
        s.seconds = s.seconds / n if n else 0.0
    return summaries


def _pct(rate: float | None) -> str:
    return "  n/a" if rate is None else f"{rate * 100:4.0f}%"


def format_report(graded: list[Graded], order: list[str]) -> str:
    summaries = summarize(graded, order)
    lines = ["", "PASS RATE PER SCENARIO (infra errors and inconclusive runs are excluded)", ""]
    lines.append(
        f"{'scenario':<24}{'runs':>5}{'pass':>6}{'fail':>6}{'infra':>7}{'inconcl.':>9}"
        f"{'pass rate':>11}{'turns':>7}{'s/run':>7}"
    )
    for sid in order:
        s = summaries[sid]
        lines.append(
            f"{sid:<24}{s.runs:>5}{s.passed:>6}{s.failed:>6}{s.infra:>7}{s.inconclusive:>9}"
            f"{_pct(s.pass_rate):>11}{s.turns:>7.1f}{s.seconds:>7.0f}"
        )
    passed = sum(s.passed for s in summaries.values())
    gradable = sum(s.gradable for s in summaries.values())
    infra = sum(s.infra for s in summaries.values())
    inconclusive = sum(s.inconclusive for s in summaries.values())
    total_rate = passed / gradable if gradable else None
    lines.append(
        f"{'ALL':<24}{gradable + infra + inconclusive:>5}{passed:>6}{gradable - passed:>6}"
        f"{infra:>7}{inconclusive:>9}{_pct(total_rate):>11}"
    )

    lines += [
        "",
        "PER-CHECK BREAKDOWN (gradable runs; checks below 100% listed, with one example)",
        "",
    ]
    for sid in order:
        s = summaries[sid]
        failing = {n: c for n, c in (s.checks or {}).items() if c[0] < c[1]}
        header = f"{sid}  ({s.passed}/{s.gradable} runs passed, {len(s.checks or {})} checks)"
        lines.append(header)
        if not failing:
            lines.append(f"  {OK_MARK} every check passed in every gradable run")
        for name, (ok, total) in sorted(failing.items(), key=lambda kv: kv[1][0] / kv[1][1]):
            detail = (s.examples or {}).get(name, "")
            lines.append(f"  {BAD_MARK} {name:<32}{ok}/{total}   e.g. {detail[:150]}")
        lines.append("")

    lines += [
        "INVARIANTS ACROSS SCENARIOS (passed/gradable runs; every scenario runs all of them)",
        "",
    ]
    header = f"{'invariant':<34}" + "".join(f"{sid[:11]:>12}" for sid in order) + f"{'ALL':>9}"
    lines.append(header)
    for name in INVARIANT_NAMES:
        cells, ok_all, total_all = [], 0, 0
        for sid in order:
            ok, total = (summaries[sid].checks or {}).get(name, [0, 0])
            ok_all, total_all = ok_all + ok, total_all + total
            mark = "" if ok == total else BAD_MARK
            cells.append(f"{mark}{ok}/{total}".rjust(12))
        lines.append(f"{name:<34}" + "".join(cells) + f"{ok_all}/{total_all}".rjust(9))

    if WARNING_NAMES:
        lines += [
            "",
            "WARNING METRICS (not pass/fail: runs where the warning fired / gradable runs)",
            "",
        ]
        for name in WARNING_NAMES:
            fired = total = 0
            per_scenario = []
            for sid in order:
                f, t = (summaries[sid].warnings or {}).get(name, [0, 0])
                fired, total = fired + f, total + t
                if t:
                    per_scenario.append(f"{sid} {f}/{t}")
            lines.append(f"{name}: {fired}/{total} runs   ({', '.join(per_scenario)})")
            example = next(
                (e for sid in order if (e := (summaries[sid].warning_examples or {}).get(name))), ""
            )
            if example:
                lines.append(f"    e.g. {example[:160]}")
    ignored = sum(s.markers_ignored for s in summaries.values())
    lines += [
        "",
        f"caller hang-up markers ignored because the line was not a farewell: {ignored}",
    ]
    return "\n".join(lines)


def format_transcript(g: Graded) -> str:
    run = g.run
    lines = [f"=== {run.scenario_id} #{run.run_index}: {g.status.upper()} ({run.outcome}) ==="]
    if run.detail:
        lines.append(f"    detail: {run.detail}")
    turn = -1
    for item in run.items:
        if item.turn != turn:
            turn = item.turn
            if turn > 0:
                lines.append(f"КЛИЕНТ: {run.caller_lines[turn - 1]}")
        if item.kind == "say":
            lines.append(f"  АГЕНТ: {item.text}")
        elif item.kind == "tool":
            args = json.dumps(item.args, ensure_ascii=False)
            mark = "ОШИБКА " if item.is_error else ("SAVED " if item.committed else "")
            lines.append(f"  [tool] {item.tool}({args[:160]}) -> {mark}{item.text[:110]}")
        elif item.kind == "failed":
            lines.append(f"  [сбой] {item.text}")
        elif item.kind == "end":
            lines.append("  [конец звонка] агент положил трубку")
    for b in run.bookings:
        lines.append(f"  saved booking: {b}")
    for m in run.messages:
        lines.append(f"  saved message: {m}")
    for r in g.failed_checks:
        lines.append(f"  {BAD_MARK} {r.name}: {r.detail}")
    for w in g.warnings:
        if not w.passed:
            lines.append(f"  ! warning {w.name}: {w.detail}")
    return "\n".join(lines)


def to_json(graded: list[Graded]) -> list[dict]:
    out = []
    for g in graded:
        run = g.run
        out.append(
            {
                "scenario": run.scenario_id,
                "run": run.run_index,
                "status": g.status,
                "outcome": run.outcome,
                "detail": run.detail,
                "seconds": round(run.seconds, 1),
                "caller_lines": run.caller_lines,
                "items": [
                    {k: v for k, v in item.__dict__.items() if v not in ("", {}, False)}
                    for item in run.items
                ],
                "bookings": [{k: str(v) for k, v in vars(b).items()} for b in run.bookings],
                "messages": [{k: str(v) for k, v in vars(m).items()} for m in run.messages],
                "markers_ignored": run.markers_ignored,
                "checks": [
                    {"name": r.name, "passed": r.passed, "detail": r.detail} for r in g.results
                ],
                "warnings": [
                    {"name": w.name, "fired": not w.passed, "detail": w.detail} for w in g.warnings
                ],
            }
        )
    return out
