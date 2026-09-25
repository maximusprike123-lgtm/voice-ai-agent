"""Plain data shared by the harness, the checks, the scenarios and the report."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from agent.business import BusinessConfig
from agent.records import Booking, CallbackMessage


@dataclass(frozen=True)
class Item:
    """One thing that happened in a call, in order."""

    turn: int  # 0 = the greeting, 1.. = the caller's utterances
    kind: str  # "say" | "tool" | "failed" | "end"
    text: str = ""  # say: the sentence; tool: its result; failed: the reason
    tool: str = ""
    args: dict = field(default_factory=dict)
    committed: bool = False
    is_error: bool = False


@dataclass
class RunResult:
    scenario_id: str
    run_index: int
    caller_lines: list[str]
    items: list[Item]
    bookings: list[Booking]
    messages: list[CallbackMessage]
    outcome: str  # "completed" | "inconclusive" | "infra_error"
    detail: str = ""
    seconds: float = 0.0
    markers_ignored: int = 0  # caller hang-up markers that came without a farewell (ignored)

    @property
    def turns(self) -> int:
        return len(self.caller_lines)

    def says(self) -> list[Item]:
        return [i for i in self.items if i.kind == "say"]

    def speech(self) -> str:
        """Everything the agent said, sentences joined by a space."""
        return " ".join(i.text for i in self.says())

    def tool_calls(self, name: str | None = None) -> list[Item]:
        return [i for i in self.items if i.kind == "tool" and (name is None or i.tool == name)]

    def ended_call(self) -> bool:
        return any(i.kind == "end" for i in self.items)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class CheckContext:
    business: BusinessConfig
    now: datetime  # the clock the call ran on


Check = Callable[[RunResult, CheckContext], CheckResult]


@dataclass(frozen=True)
class Scenario:
    id: str
    description: str
    persona: str  # who the caller is and what they want
    facts: str  # what they know and answer when asked
    behavior: str  # how they react to specific situations
    caller_phone: str | None  # caller ID the agent sees (None = hidden number)
    checks: tuple[Check, ...]  # scenario-specific; the invariants are added for every scenario
    max_turns: int = 16
