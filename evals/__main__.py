"""Run the scenario evaluations. Calls real LLMs and costs (a little) money: on demand only.

    .venv/bin/python -m evals --dry-run                    # plan + cost estimate, no LLM calls
    .venv/bin/python -m evals                              # every scenario, 5 runs each
    .venv/bin/python -m evals --scenarios sunday_closed,price_only --runs 3 --show-failures

The agent under test is whatever .env configures (with LLM_EXTRA_BODY etc.). The simulated
caller is a DIFFERENT, cheap model family: the first of CALLER_MODEL_CANDIDATES that works for
this account (privacy settings included), or --caller-model, or, only as a last resort, the
agent's own model. Results, transcripts and a JSON dump go to data/evals/<timestamp>/.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

from agent.business import load_business_config
from agent.llm import OpenAICompatibleLLMClient
from agent.settings import Settings, get_settings
from evals.checks import grade
from evals.cost import Price, UsageMeter, estimate_cost, fetch_prices
from evals.harness import run_once
from evals.model import CheckContext, Scenario
from evals.report import Graded, format_report, format_transcript, to_json
from evals.scenarios import BY_ID, SCENARIOS

# Cheap, non-DeepSeek families, best first (measured: all pass this account's privacy filters).
CALLER_MODEL_CANDIDATES = (
    "google/gemini-2.5-flash-lite",
    "openai/gpt-4.1-nano",
    "mistralai/mistral-small-3.2-24b-instruct",
    "meta-llama/llama-3.3-70b-instruct",
)
CALLER_TEMPERATURE = 0.7
CALLER_MAX_TOKENS = 150
DEFAULT_RUNS = 5
DEFAULT_CONCURRENCY = 4
DEFAULT_MAX_COST = 2.0
FAILURES_SHOWN_PER_SCENARIO = 3


def fixed_now(settings: Settings) -> datetime:
    """Friday 2026-09-25 15:00 in the business time zone (see scenarios.py)."""
    return datetime(2026, 9, 25, 15, 0, tzinfo=settings.tz)


async def pick_caller_model(
    settings: Settings,
    candidates: tuple[str, ...] = CALLER_MODEL_CANDIDATES,
    client: httpx.AsyncClient | None = None,
) -> tuple[str, str]:
    """The first candidate from another family that answers a tiny request on this account."""
    agent_family = settings.llm_model.split("/")[0]
    owns = client is None
    client = client or httpx.AsyncClient(
        base_url=settings.llm_base_url,
        headers={"Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}"},
        timeout=30,
    )
    skipped = []
    try:
        for model in candidates:
            if model.split("/")[0] == agent_family:
                continue
            body = {"model": model, "messages": [{"role": "user", "content": "Скажи: да"}]}
            try:
                response = await client.post("/chat/completions", json={**body, "max_tokens": 8})
                if response.status_code != 200:
                    skipped.append(f"{model}: HTTP {response.status_code}")
                    continue
                content = response.json()["choices"][0]["message"].get("content") or ""
                if content.strip():
                    return model, f"first available of {len(candidates)} candidates" + (
                        f" (skipped: {', '.join(skipped)})" if skipped else ""
                    )
                skipped.append(f"{model}: empty answer")
            except (httpx.HTTPError, KeyError, IndexError, ValueError):
                skipped.append(f"{model}: unusable")
    finally:
        if owns:
            await client.aclose()
    return settings.llm_model, f"FALLBACK to the agent's own model, no candidate worked ({skipped})"


def select_scenarios(names: str | None) -> list[Scenario]:
    if not names:
        return list(SCENARIOS)
    chosen = []
    for name in (n.strip() for n in names.split(",") if n.strip()):
        if name not in BY_ID:
            raise SystemExit(f"unknown scenario {name!r}; available: {', '.join(BY_ID)}")
        chosen.append(BY_ID[name])
    return chosen


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evals", description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenarios", help="comma-separated ids (default: all)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="runs per scenario")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--caller-model", help="override the simulated caller's model")
    parser.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST, help="USD; abort above")
    parser.add_argument("--dry-run", action="store_true", help="print plan and estimate only")
    parser.add_argument("--show-failures", action="store_true", help="print failing transcripts")
    parser.add_argument("--min-pass-rate", type=float, help="exit 1 if a scenario is below this")
    parser.add_argument("--out", type=Path, help="results directory (default data/evals/<time>)")
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    return parser


async def sweep(
    scenarios: list[Scenario],
    runs: int,
    *,
    settings: Settings,
    agent_llm,
    caller_llm,
    concurrency: int,
    stop_check,
) -> list[Graded]:
    business = load_business_config(settings.business_config_path)
    now = fixed_now(settings)
    ctx = CheckContext(business, now)
    clock = lambda: now  # noqa: E731
    jobs = [(scenario, i) for i in range(runs) for scenario in scenarios]  # round-robin
    semaphore = asyncio.Semaphore(concurrency)
    graded: list[Graded] = []
    finished = 0

    async def one(scenario: Scenario, index: int) -> None:
        nonlocal finished
        async with semaphore:
            if stop_check():
                return
            result = await run_once(
                scenario,
                index,
                settings=settings,
                agent_llm=agent_llm,
                caller_llm=caller_llm,
                clock=clock,
            )
            g = Graded(result, grade(result, scenario.checks, ctx))
            graded.append(g)
            finished += 1
            failed = ", ".join(r.name for r in g.failed_checks) if g.status == "fail" else ""
            extra = (
                f"  failed: {failed}" if failed else (f"  {g.run.detail}" if g.run.detail else "")
            )
            print(
                f"[{finished}/{len(jobs)}] {scenario.id} #{index}: {g.status.upper()} "
                f"({g.run.turns} turns, {g.run.seconds:.0f}s){extra}",
                flush=True,
            )

    await asyncio.gather(*(one(s, i) for s, i in jobs))
    return graded


def role_prices(prices: dict[str, Price], agent_model: str, caller_model: str) -> dict[str, Price]:
    out = {}
    if agent_model in prices:
        out["agent"] = prices[agent_model]
    if caller_model in prices:
        out["caller"] = prices[caller_model]
    return out


async def amain(args: argparse.Namespace) -> int:
    settings = get_settings()
    scenarios = select_scenarios(args.scenarios)
    total_runs = len(scenarios) * args.runs
    agent_model = settings.llm_model

    if args.caller_model:
        caller_model, note = args.caller_model, "from --caller-model"
    elif args.dry_run:
        caller_model, note = CALLER_MODEL_CANDIDATES[0], "would be probed at run time"
    else:
        caller_model, note = await pick_caller_model(settings)

    prices = role_prices(
        await fetch_prices(settings.llm_base_url, [agent_model, caller_model]),
        agent_model,
        caller_model,
    )
    estimate = estimate_cost(total_runs, prices)

    print(f"agent under test : {agent_model}  (extra_body={settings.llm_extra_body})")
    print(f"simulated caller : {caller_model}  [{note}]")
    print(
        f"plan             : {len(scenarios)} scenarios x {args.runs} runs = {total_runs} calls, "
        f"concurrency {args.concurrency}"
    )
    if prices:
        print(
            "listed prices    : "
            + ", ".join(
                f"{role} ${p[0]:.3f}/${p[1]:.3f} per M tokens" for role, p in prices.items()
            )
        )
    print(
        "estimated cost   : "
        + (
            f"about ${estimate:.2f} (max allowed ${args.max_cost:.2f})"
            if estimate is not None
            else "unknown (no price list for this backend)"
        )
    )
    if args.dry_run:
        return 0
    if estimate is not None and estimate > args.max_cost:
        print(f"\nESTIMATE ${estimate:.2f} EXCEEDS --max-cost ${args.max_cost:.2f}: not running.")
        return 2

    meter = UsageMeter()
    agent_llm = OpenAICompatibleLLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=agent_model,
        reasoning_effort=settings.llm_reasoning_effort,
        extra_body=settings.llm_extra_body,
        usage_hook=meter.hook("agent"),
        timeout_seconds=settings.llm_timeout_seconds,
    )
    same_model = caller_model == agent_model
    caller_llm = OpenAICompatibleLLMClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key.get_secret_value(),
        model=caller_model,
        reasoning_effort=settings.llm_reasoning_effort if same_model else None,
        extra_body={"temperature": CALLER_TEMPERATURE, "max_tokens": CALLER_MAX_TOKENS},
        usage_hook=meter.hook("caller"),
        timeout_seconds=30,
    )

    over_budget = {"hit": False}

    def stop_check() -> bool:
        cost = meter.total_cost(prices)
        if cost is not None and cost > args.max_cost and not over_budget["hit"]:
            over_budget["hit"] = True
            print(
                f"\n!! spent ${cost:.2f} > --max-cost ${args.max_cost:.2f}: stopping early",
                flush=True,
            )
        return over_budget["hit"]

    print()
    started = datetime.now()
    graded = await sweep(
        scenarios,
        args.runs,
        settings=settings,
        agent_llm=agent_llm,
        caller_llm=caller_llm,
        concurrency=args.concurrency,
        stop_check=stop_check,
    )
    graded.sort(key=lambda g: ([s.id for s in scenarios].index(g.run.scenario_id), g.run.run_index))
    order = [s.id for s in scenarios]

    report = format_report(graded, order)
    print(report)
    print(cost_report(meter, prices, order))

    out_dir = args.out or Path("data") / "evals" / started.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    (out_dir / "results.json").write_text(
        json.dumps(to_json(graded), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out_dir / "transcripts.txt").write_text(
        "\n\n".join(format_transcript(g) for g in graded), encoding="utf-8"
    )
    print(f"\nsaved: {out_dir}/ (report.txt, results.json, transcripts.txt)")

    if args.show_failures:
        shown: dict[str, int] = {}
        for g in graded:
            if g.status != "pass" and shown.get(g.run.scenario_id, 0) < FAILURES_SHOWN_PER_SCENARIO:
                shown[g.run.scenario_id] = shown.get(g.run.scenario_id, 0) + 1
                print("\n" + format_transcript(g))

    await agent_llm.aclose()
    await caller_llm.aclose()

    if args.min_pass_rate is not None:
        from evals.report import summarize

        low = [
            s.scenario_id
            for s in summarize(graded, order).values()
            if s.pass_rate is not None and s.pass_rate < args.min_pass_rate
        ]
        if low:
            print(f"\nBELOW --min-pass-rate {args.min_pass_rate:.0%}: {', '.join(low)}")
            return 1
    return 0


def _usd(cost: float | None) -> str:
    return "n/a" if cost is None else f"${cost:.3f}"


def cost_report(meter: UsageMeter, prices: dict[str, Price], order: list[str]) -> str:
    lines = ["", "TOKENS AND APPROXIMATE COST (listed prices; the serving provider may differ)", ""]
    lines.append(f"{'':<24}{'requests':>9}{'prompt tok':>12}{'compl. tok':>12}{'~USD':>9}")
    for role in ("agent", "caller"):
        u = meter.by_role[role]
        cost = _usd(u.cost(prices.get(role)))
        lines.append(f"{role:<24}{u.requests:>9}{u.prompt:>12,}{u.completion:>12,}{cost:>9}")
    total = _usd(meter.total_cost(prices))
    lines.append(f"{'TOTAL':<24}{'':>9}{'':>12}{'':>12}{total:>9}")
    lines += ["", "per scenario (agent + caller tokens):"]
    for sid in order:
        roles = meter.by_scenario.get(sid, {})
        prompt = sum(u.prompt for u in roles.values())
        completion = sum(u.completion for u in roles.values())
        cost = sum((u.cost(prices.get(r)) or 0.0) for r, u in roles.items())
        lines.append(f"  {sid:<24}{prompt:>10,} in {completion:>8,} out   ~${cost:.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        for s in SCENARIOS:
            print(f"{s.id:<24}{s.description}")
        return 0
    if args.runs < 1 or args.concurrency < 1:
        print("--runs and --concurrency must be at least 1", file=sys.stderr)
        return 2
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
