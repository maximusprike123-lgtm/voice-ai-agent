"""Token accounting and a rough dollar estimate for a sweep."""

import contextvars
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

# Which run is executing ("scenario#index"), so usage reported by a client that is shared by
# concurrent runs can be attributed to the right scenario. Set by the harness per run.
current_run: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_run", default=None
)

# The upstream provider named by the backend for the request being served (set by the usage
# hook, read by the request telemetry, both inside the same task).
last_provider: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "last_provider", default=None
)

Price = tuple[float, float]  # USD per million tokens: (prompt, completion)

# Rough size of one run by scenario: {role: (prompt tokens, completion tokens)}. An agent turn
# re-sends the ~2k-token prompt plus the history, so a long booking call costs about 2x a
# medium one and 4x a one-question call. Measured in the 2026-09-25 sweep of 50 runs (actual
# $0.156 vs estimate $0.167): booking runs ~28k tokens in total (agent + caller), medium ~13k,
# short ~7k. Update when the prompt or the scenarios change.
BOOKING_RUN = {"agent": (24_000, 450), "caller": (4_000, 100)}
MEDIUM_RUN = {"agent": (11_500, 250), "caller": (1_900, 80)}
SHORT_RUN = {"agent": (6_500, 100), "caller": (900, 40)}
ESTIMATED_TOKENS_PER_RUN: dict[str, dict[str, tuple[int, int]]] = {
    "happy_path_booking": BOOKING_RUN,
    "approximate_time": BOOKING_RUN,
    "changes_mind": BOOKING_RUN,
    "hidden_caller_id": BOOKING_RUN,
    "service_not_listed": BOOKING_RUN,
    "sunday_closed": BOOKING_RUN,
    "question_outside_faq": MEDIUM_RUN,
    "rude_offtopic": MEDIUM_RUN,
    "address_only": MEDIUM_RUN,
    "price_only": SHORT_RUN,
}
DEFAULT_RUN_ESTIMATE = BOOKING_RUN  # for a scenario that has no entry (be pessimistic)


@dataclass
class Usage:
    prompt: int = 0
    completion: int = 0
    requests: int = 0

    def add(self, usage: dict) -> None:
        self.prompt += int(usage.get("prompt_tokens") or 0)
        self.completion += int(usage.get("completion_tokens") or 0)
        self.requests += 1

    def cost(self, price: Price | None) -> float | None:
        if price is None:
            return None
        return (self.prompt * price[0] + self.completion * price[1]) / 1_000_000


@dataclass
class UsageMeter:
    """Collects usage from the two LLM clients (roles: 'agent', 'caller')."""

    by_role: dict[str, Usage] = field(default_factory=lambda: {"agent": Usage(), "caller": Usage()})
    by_scenario: dict[str, dict[str, Usage]] = field(default_factory=dict)

    def hook(self, role: str) -> Callable[[dict], None]:
        def record(usage: dict) -> None:
            self.by_role[role].add(usage)
            if usage.get("provider"):
                last_provider.set(usage["provider"])
            key = current_run.get()
            if key:
                scenario = key.split("#")[0]
                self.by_scenario.setdefault(scenario, {}).setdefault(role, Usage()).add(usage)

        return record

    def total_cost(self, prices: dict[str, Price]) -> float | None:
        """Dollars so far, or None if a role's price is unknown."""
        total = 0.0
        for role, usage in self.by_role.items():
            cost = usage.cost(prices.get(role))
            if cost is None:
                return None
            total += cost
        return total


def estimate_cost(scenario_ids: Iterable[str], prices: dict[str, Price]) -> float | None:
    """Estimated dollars for one run of each listed scenario id (repeat an id per run)."""
    total = 0.0
    for scenario_id in scenario_ids:
        sizes = ESTIMATED_TOKENS_PER_RUN.get(scenario_id, DEFAULT_RUN_ESTIMATE)
        for role, (tokens_in, tokens_out) in sizes.items():
            price = prices.get(role)
            if price is None:
                return None
            total += (tokens_in * price[0] + tokens_out * price[1]) / 1_000_000
    return total


async def fetch_prices(
    base_url: str, models: Iterable[str], client: httpx.AsyncClient | None = None
) -> dict[str, Price]:
    """Listed prices per million tokens from OpenRouter's public model list ({} for any other
    backend or on failure). The provider that serves a request may charge a little differently,
    so treat the result as approximate."""
    if urlparse(base_url).hostname != "openrouter.ai":
        return {}
    wanted = set(models)
    owns = client is None
    client = client or httpx.AsyncClient(timeout=15)
    try:
        response = await client.get(f"{base_url.rstrip('/')}/models")
        response.raise_for_status()
        data = response.json().get("data", [])
    except (httpx.HTTPError, ValueError):
        return {}
    finally:
        if owns:
            await client.aclose()
    prices: dict[str, Price] = {}
    for entry in data:
        if entry.get("id") in wanted:
            pricing = entry.get("pricing") or {}
            try:
                prices[entry["id"]] = (
                    float(pricing["prompt"]) * 1_000_000,
                    float(pricing["completion"]) * 1_000_000,
                )
            except (KeyError, TypeError, ValueError):
                continue
    return prices
