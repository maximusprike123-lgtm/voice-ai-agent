"""Run ONE simulated call: caller LLM <-> the real CallSession/ToolRegistry/SQLite."""

import asyncio
import tempfile
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from agent.app import open_runtime
from agent.dialogue import EndCall, Say, SentenceBlocked, ToolCallsDropped, ToolResult
from agent.llm import LLMClient
from agent.session import TurnFailed
from agent.settings import Settings
from evals.caller import CallerError, SimulatedCaller
from evals.cost import current_run
from evals.model import Item, RunResult, Scenario


async def run_once(
    scenario: Scenario,
    run_index: int,
    *,
    settings: Settings,
    agent_llm: LLMClient,
    caller_llm: LLMClient,
    clock: Callable[[], datetime],
    db_dir: Path | None = None,
) -> RunResult:
    """One call from greeting to hang-up. Never raises: failures become the run's `outcome`."""
    current_run.set(f"{scenario.id}#{run_index}")
    started = time.monotonic()
    items: list[Item] = []
    caller_lines: list[str] = []
    outcome, detail = "completed", ""
    bookings, messages = [], []
    markers_ignored = 0
    farewells_deferred = 0

    with tempfile.TemporaryDirectory() as tmp:
        db_path = (db_dir or Path(tmp)) / f"{scenario.id}_{run_index}.db"
        try:
            async with open_runtime(settings, db_path=db_path, clock=clock, llm=agent_llm) as rt:
                session = rt.new_call(scenario.caller_phone)
                caller = SimulatedCaller(caller_llm, scenario)

                greeting = session.greet()
                items += [Item(0, "say", text=s.text) for s in greeting]
                heard = " ".join(s.text for s in greeting)

                try:
                    for turn in range(1, scenario.max_turns + 1):
                        try:
                            line, done = await caller.next_utterance(heard)
                        except CallerError as exc:
                            outcome, detail = "inconclusive", str(exc)
                            break
                        caller_lines.append(line)

                        said: list[str] = []
                        async for event in session.handle(line):
                            if isinstance(event, Say):
                                items.append(Item(turn, "say", text=event.text))
                                said.append(event.text)
                            elif isinstance(event, ToolResult):
                                items.append(_tool_item(turn, event))
                            elif isinstance(event, SentenceBlocked):
                                items.append(
                                    Item(turn, "blocked", text=event.text, rule=event.rule)
                                )
                            elif isinstance(event, ToolCallsDropped):
                                items.append(
                                    Item(
                                        turn,
                                        "dropped",
                                        text=", ".join(event.tools),
                                        rule=event.rule,
                                    )
                                )
                            elif isinstance(event, TurnFailed):
                                items.append(Item(turn, "failed", text=event.reason))
                                outcome, detail = "infra_error", event.reason
                            elif isinstance(event, EndCall):
                                items.append(Item(turn, "end"))
                        heard = " ".join(said)

                        if outcome == "infra_error":
                            break  # the run is void; do not spend more tokens on it
                        if session.ended:
                            break
                        if done:
                            if said and said[-1].strip().endswith("?"):
                                # The caller said goodbye, but the agent replied with a question
                                # (e.g. the read-back «Всё верно?»): a real caller would answer
                                # it before hanging up, so let the caller continue.
                                farewells_deferred += 1
                                continue
                            break
                    else:
                        outcome, detail = "inconclusive", f"no end after {scenario.max_turns} turns"
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # keep what was saved so far, report the failure
                    outcome, detail = "infra_error", f"{type(exc).__name__}: {exc}"

                markers_ignored = caller.markers_ignored
                bookings = [s.booking for s in await rt.store.list_bookings()]
                messages = [s.message for s in await rt.store.list_messages()]
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # anything unexpected is reported, never raised into the sweep
            outcome, detail = "infra_error", f"{type(exc).__name__}: {exc}"

    return RunResult(
        scenario_id=scenario.id,
        run_index=run_index,
        caller_lines=caller_lines,
        items=items,
        bookings=bookings,
        messages=messages,
        outcome=outcome,
        detail=detail,
        seconds=time.monotonic() - started,
        markers_ignored=markers_ignored,
        farewells_deferred=farewells_deferred,
    )


def _tool_item(turn: int, event: ToolResult) -> Item:
    import json

    try:
        args = json.loads(event.call.arguments or "{}")
    except json.JSONDecodeError:
        args = {"_raw": event.call.arguments}
    return Item(
        turn,
        "tool",
        text=event.result,
        tool=event.call.name,
        args=args if isinstance(args, dict) else {"_raw": args},
        committed=event.committed,
        is_error=event.result.startswith("ОШИБКА:"),
    )
