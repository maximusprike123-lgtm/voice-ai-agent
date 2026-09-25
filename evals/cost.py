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

Price = tuple[float, float]  # USD per million tokens: (prompt, completion)

# Rough size of one run, measured from CLI runs: an agent turn re-sends the ~2k-token prompt.
ESTIMATED_TOKENS_PER_RUN = {"agent": (25_000, 700), "caller": (9_000, 500)}


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


def estimate_cost(runs: int, prices: dict[str, Price]) -> float | None:
    total = 0.0
    for role, (tokens_in, tokens_out) in ESTIMATED_TOKENS_PER_RUN.items():
        price = prices.get(role)
        if price is None:
            return None
        total += runs * (tokens_in * price[0] + tokens_out * price[1]) / 1_000_000
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
