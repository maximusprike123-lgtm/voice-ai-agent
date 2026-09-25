"""Time-to-first-token study for the configured OpenRouter model, per routing configuration.

Not part of `pytest`: real requests, a few cents of API credit.

    .venv/bin/python scripts/latency_study.py                      # 40 requests per configuration
    .venv/bin/python scripts/latency_study.py --n 50 --providers Fireworks,DeepInfra,Together

Configurations compared (all with the real system prompt, the real tool schemas and thinking
off, i.e. what a phone call sends):
  default          no routing preferences: OpenRouter picks the provider
  sort=latency     {"provider": {"sort": "latency"}}
  pin:<Provider>   {"provider": {"order": [<Provider>], "allow_fallbacks": false}}

Each round sends ONE request per configuration at the same moment (concurrently), so
time-of-day and network conditions hit every configuration equally. The volatile part of the
prompt (time, caller) and the caller's question change from request to request, like real
calls, while the static prefix stays identical (so provider-side prompt caching can help,
exactly as it would in production).

Two times are reported per request, both in seconds from sending the request:
  TTFT   the first useful output (a text delta or the first tool-call fragment);
  event  the first thing DialogueEngine's stall timeout can see: a text delta, or for a
         tool-call reply the moment the tool call is COMPLETE (LLMClient only emits a tool call
         once its arguments have fully arrived), so it is >= TTFT and can be much later.
Failures and timeouts are counted as "over the threshold" and reported separately.
With few samples p99 is simply the maximum: read p90 and the over-4s share instead.
Raw per-request results go to a JSON file (default data/, which is gitignored).
"""

import argparse
import asyncio
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent.business import load_business_config  # noqa: E402
from agent.llm import Message, OpenAICompatibleLLMClient, Role  # noqa: E402
from agent.prompt import build_system_prompt  # noqa: E402
from agent.settings import get_settings  # noqa: E402
from agent.tools import build_tool_specs  # noqa: E402

THRESHOLD_SECONDS = 4.0  # the engine's first-event timeout
SECOND_THRESHOLD_SECONDS = 2.0
# Providers eligible for this account. NOTE: the account's privacy guardrails (zero data
# retention, no training on prompts) remove deepseek, alibaba, streamlake, gmicloud and
# atlas-cloud from routing, so pinning "DeepSeek" fails with HTTP 404.
DEFAULT_PROVIDERS = ("Fireworks", "DeepInfra", "Together")
QUESTIONS = [
    "Сколько стоит полировка кузова?",
    "Где вы находитесь и до скольки работаете?",
    "Запишите меня на завтра на керамику.",
    "Можно оставить машину на ночь?",
    "Какие у вас способы оплаты?",
    "А сколько по времени занимает химчистка салона?",
    # Likely to be answered with a tool call (arguments must be generated before anything is
    # visible to the engine):
    "Запишите меня на полировку кузова на завтра днём: Игорь, телефон 8 916 123 45 67, "
    "Тойота Камри.",
    "Передайте администратору, что я хочу обсудить скидку для постоянных клиентов.",
]


@dataclass
class Sample:
    config: str
    round: int
    ttft: float | None  # None: the request failed or produced nothing
    event: float | None  # first client-visible event (see the module docstring)
    total: float
    provider: str | None
    error: str | None


def configurations(providers: list[str]) -> dict[str, dict | None]:
    configs: dict[str, dict | None] = {
        "default": None,
        "sort=latency": {"provider": {"sort": "latency"}},
    }
    for name in providers:
        configs[f"pin:{name}"] = {"provider": {"order": [name], "allow_fallbacks": False}}
    return configs


def percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile of an already sorted, non-empty list."""
    rank = max(1, math.ceil(p / 100 * len(sorted_values)))
    return sorted_values[rank - 1]


def share_over(ok: list[float], failed: int, limit: float) -> float:
    """Percent of all requests slower than `limit`, counting every failure as slower."""
    return 100 * (sum(v > limit for v in ok) + failed) / (len(ok) + failed)


async def one_request(
    http: httpx.AsyncClient,
    config: str,
    extra_body: dict | None,
    round_no: int,
    settings,
    business,
    timeout: float,
) -> Sample:
    now = datetime.now(settings.tz) + timedelta(minutes=round_no)
    phone = f"+7999{round_no:07d}"
    messages = [
        Message(Role.SYSTEM, build_system_prompt(business, now, phone)),
        Message(Role.USER, QUESTIONS[round_no % len(QUESTIONS)]),
    ]
    # Build the body with the real client, so the study sends exactly what production sends
    # (model, reasoning_effort, extra_body passthrough, tools).
    client = OpenAICompatibleLLMClient(
        base_url=settings.llm_base_url,
        api_key="unused",
        model=settings.llm_model,
        reasoning_effort=settings.llm_reasoning_effort,
        extra_body=extra_body,
    )
    body = client._build_body(messages, build_tool_specs(business))

    started = time.monotonic()
    ttft = event = provider = error = None
    saw_tool_call = False
    try:
        async with http.stream("POST", "/chat/completions", json=body, timeout=timeout) as response:
            if response.status_code != 200:
                text = (await response.aread()).decode("utf-8", "replace")
                error = f"HTTP {response.status_code}: {text[:120]}"
            else:
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    provider = provider or chunk.get("provider")
                    choice = (chunk.get("choices") or [{}])[0]
                    delta = choice.get("delta") or {}
                    now_s = time.monotonic() - started
                    saw_tool_call = saw_tool_call or bool(delta.get("tool_calls"))
                    if ttft is None and (delta.get("content") or delta.get("tool_calls")):
                        ttft = now_s
                    if event is None and (
                        delta.get("content") or (saw_tool_call and choice.get("finish_reason"))
                    ):
                        event = now_s
    except httpx.TimeoutException:
        error = f"timeout after {timeout:.0f}s"
    except httpx.HTTPError as exc:
        error = type(exc).__name__
    if error is None and ttft is None:
        error = "no output"
    return Sample(config, round_no, ttft, event, time.monotonic() - started, provider, error)


def report(samples: list[Sample], configs: list[str]) -> None:
    print(
        f"\n{'configuration':<22}{'ok/n':>7}{'p50':>7}{'p90':>7}{'p99':>7}{'max':>7}"
        f"{'mean':>7}{'>2s':>7}{'>4s':>7}   providers seen"
    )
    for config in configs:
        mine = [s for s in samples if s.config == config]
        ok = sorted(s.ttft for s in mine if s.ttft is not None)
        failed = len(mine) - len(ok)

        seen: dict[str, int] = {}
        for s in mine:
            seen[s.provider or "?"] = seen.get(s.provider or "?", 0) + 1
        providers = ", ".join(f"{k} x{v}" for k, v in sorted(seen.items(), key=lambda kv: -kv[1]))
        if ok:
            stats = (
                f"{percentile(ok, 50):7.2f}{percentile(ok, 90):7.2f}{percentile(ok, 99):7.2f}"
                f"{ok[-1]:7.2f}{statistics.mean(ok):7.2f}"
            )
        else:
            stats = " " * 35
        over2 = share_over(ok, failed, SECOND_THRESHOLD_SECONDS)
        over4 = share_over(ok, failed, THRESHOLD_SECONDS)
        label = f"{config:<22}{len(ok):>3}/{len(mine):<3}"
        print(f"{label}{stats}{over2:6.0f}%{over4:6.0f}%   {providers}")

    print("\n(seconds; '>2s' and '>4s' are the share of ALL requests, failures included)")

    print("\nfirst client-visible event (what the engine's stall timeout waits for):")
    print(f"{'configuration':<22}{'p50':>7}{'p90':>7}{'max':>7}{'>2s':>7}{'>4s':>7}")
    for config in configs:
        mine = [s for s in samples if s.config == config]
        ok = sorted(s.event for s in mine if s.event is not None)
        failed = len(mine) - len(ok)
        if not ok:
            print(f"{config:<22}  no successful requests")
            continue
        over2 = share_over(ok, failed, SECOND_THRESHOLD_SECONDS)
        over4 = share_over(ok, failed, THRESHOLD_SECONDS)
        print(
            f"{config:<22}{percentile(ok, 50):7.2f}{percentile(ok, 90):7.2f}{ok[-1]:7.2f}"
            f"{over2:6.0f}%{over4:6.0f}%"
        )
    tool_reply = [s for s in samples if s.ttft is not None and s.event and s.event - s.ttft > 0.3]
    if tool_reply:
        gaps = sorted(s.event - s.ttft for s in tool_reply)
        print(
            f"\n{len(tool_reply)} replies were tool calls: the event arrives on average "
            f"{statistics.mean(gaps):.2f}s (max {gaps[-1]:.2f}s) after the first token."
        )
    errors = [s for s in samples if s.error]
    if errors:
        print(f"\nfailures ({len(errors)}):")
        for s in errors[:12]:
            print(f"  {s.config} round {s.round}: {s.error}")

    default = [s for s in samples if s.config == "default" and s.ttft is not None]
    by_provider: dict[str, list[float]] = {}
    for s in default:
        by_provider.setdefault(s.provider or "?", []).append(s.ttft)
    if len(by_provider) > 1:
        print("\ndefault routing, by the provider that actually served the request:")
        for name, values in sorted(by_provider.items(), key=lambda kv: -len(kv[1])):
            values.sort()
            print(
                f"  {name:<18} n={len(values):<3} p50 {percentile(values, 50):5.2f}s  "
                f"max {values[-1]:5.2f}s"
            )


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=40, help="requests per configuration (30-50)")
    parser.add_argument(
        "--providers",
        default=",".join(DEFAULT_PROVIDERS),
        help="comma-separated providers to pin (exact OpenRouter names)",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request timeout, seconds")
    parser.add_argument("--pause", type=float, default=0.5, help="pause between rounds, seconds")
    parser.add_argument("--out", type=Path, help="raw results (default data/latency_study_*.json)")
    args = parser.parse_args()

    settings = get_settings()
    if settings.llm_reasoning_effort != "none":
        print("LLM_REASONING_EFFORT must be 'none' for this study (thinking off, like a call).")
        return 2
    business = load_business_config(settings.business_config_path)
    configs = configurations([p.strip() for p in args.providers.split(",") if p.strip()])
    out = args.out or Path("data") / f"latency_study_{datetime.now():%Y%m%d_%H%M%S}.json"

    print(f"model: {settings.llm_model}   backend: {settings.llm_base_url}")
    print(
        f"{args.n} rounds x {len(configs)} configurations = {args.n * len(configs)} requests, "
        "sent concurrently per round"
    )

    samples: list[Sample] = []
    async with httpx.AsyncClient(
        base_url=settings.llm_base_url,
        headers={"Authorization": f"Bearer {settings.llm_api_key.get_secret_value()}"},
    ) as http:
        for round_no in range(args.n):
            batch = await asyncio.gather(
                *(
                    one_request(http, name, extra, round_no, settings, business, args.timeout)
                    for name, extra in configs.items()
                )
            )
            samples.extend(batch)
            done = round_no + 1
            if done % 10 == 0 or done == args.n:
                print(f"  {done}/{args.n} rounds done", flush=True)
            await asyncio.sleep(args.pause)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([asdict(s) for s in samples], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    report(samples, list(configs))
    print(f"\nraw results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
