"""Offline tests for the eval harness: simulated caller, one-call runner, cost accounting,
report and the command line. No network: fake LLMs stand in for both sides of the call."""

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest

from agent.llm import LLMError, Message, Role, StreamEnd, TextDelta, ToolCall, ToolCallEvent
from agent.settings import Settings
from evals import __main__ as evals_main
from evals.caller import (
    END_MARKER,
    FALLBACK_FAREWELL,
    CallerError,
    SimulatedCaller,
    build_persona_prompt,
    clean_reply,
)
from evals.checks import INVARIANT_NAMES
from evals.cost import (
    Usage,
    UsageMeter,
    current_run,
    estimate_cost,
    fetch_prices,
)
from evals.harness import run_once
from evals.model import CheckResult, Item, RunResult
from evals.report import Graded, format_report, format_transcript, summarize, to_json
from evals.scenarios import BY_ID, SCENARIOS

MOSCOW = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 9, 25, 15, 0, tzinfo=MOSCOW)
REPO_CONFIG = Path(__file__).parent.parent / "config" / "business.yaml"


def make_settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        llm_base_url="https://openrouter.ai/api/v1",
        llm_api_key="key",
        llm_model="deepseek/deepseek-v4.1-flash",
        telegram_bot_token="1:a",
        telegram_chat_id="1",
        business_config_path=REPO_CONFIG,
        db_path=tmp_path / "unused.db",
    )


class ScriptedLLM:
    """Plays one scripted reply per stream() call. An Exception in the script is raised."""

    def __init__(self, *scripts):
        self._scripts = list(scripts)
        self.calls: list[list[Message]] = []

    async def stream(self, messages, tools=None):
        self.calls.append(list(messages))
        script = self._scripts.pop(0)
        if isinstance(script, Exception):
            raise script
        for event in script:
            yield event


def text(*chunks):
    return [*(TextDelta(c) for c in chunks), StreamEnd("stop")]


def tool_round(name, args=None):
    call = ToolCall(f"call_{name}", name, json.dumps(args or {}, ensure_ascii=False))
    return [ToolCallEvent(call), StreamEnd("tool_calls")]


BOOKING_ARGS = {
    "name": "Игорь",
    "phone": "8 916 123 45 67",
    "car": "Тойота Камри",
    "service_id": "polishing",
    "preferred_date": "2026-09-26",
    "preferred_time": "14:00",
}


# --- The simulated caller -------------------------------------------------------------------------


def test_the_persona_prompt_contains_the_scenarios_parts_and_the_end_marker():
    scenario = BY_ID["hidden_caller_id"]
    prompt = build_persona_prompt(scenario)

    for part in (scenario.persona, scenario.facts, scenario.behavior):
        assert part.strip() in prompt
    assert END_MARKER in prompt and "ТОЛЬКО репликой клиента" in prompt


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.id)
def test_every_persona_prompt_builds_without_leftover_placeholders(scenario):
    prompt = build_persona_prompt(scenario)
    assert "{" not in prompt and "}" not in prompt


@pytest.mark.parametrize(
    ("raw", "line", "done"),
    [
        ("Хочу записаться на полировку.", "Хочу записаться на полировку.", False),
        ("  Клиент: Меня зовут Игорь.  ", "Меня зовут Игорь.", False),
        ("«Да, всё верно.»", "Да, всё верно.", False),
        ('"Нет, спасибо. До свидания." [КОНЕЦ]', "Нет, спасибо. До свидания.", True),
        ("Всё, надоели. [КОНЕЦ]", "Всё, надоели.", True),
        ("[КОНЕЦ]", FALLBACK_FAREWELL, True),
        ("До свидания! [КОНЕЦ", "До свидания!", True),
    ],
)
def test_clean_reply(raw, line, done):
    assert clean_reply(raw) == (line, done)


async def test_the_caller_hears_only_what_the_agent_said_and_keeps_its_own_lines():
    llm = ScriptedLLM(text("Хочу записаться."), text("Меня зовут Игорь."))
    caller = SimulatedCaller(llm, BY_ID["happy_path_booking"])

    first = await caller.next_utterance("Здравствуйте! Чем могу помочь?")
    second = await caller.next_utterance("Как вас зовут?")

    assert first == ("Хочу записаться.", False) and second == ("Меня зовут Игорь.", False)
    roles = [(m.role, m.content) for m in llm.calls[1]]
    assert roles[0][0] is Role.SYSTEM
    assert roles[1:] == [
        (Role.USER, "Здравствуйте! Чем могу помочь?"),  # the agent, from the caller's side
        (Role.ASSISTANT, "Хочу записаться."),  # the caller's own earlier line
        (Role.USER, "Как вас зовут?"),
    ]


async def test_the_end_marker_is_reported_and_stripped():
    caller = SimulatedCaller(
        ScriptedLLM(text("Спасибо, до свидания. [КОНЕЦ]")), BY_ID["price_only"]
    )
    assert await caller.next_utterance("Чем помочь?") == ("Спасибо, до свидания.", True)


async def test_an_empty_or_failed_generation_is_retried_once_then_reported():
    retried = SimulatedCaller(ScriptedLLM(text(""), text("Хочу записаться.")), BY_ID["price_only"])
    assert (await retried.next_utterance("Чем помочь?"))[0] == "Хочу записаться."

    failing = SimulatedCaller(ScriptedLLM(LLMError("x"), LLMError("y")), BY_ID["price_only"])
    with pytest.raises(CallerError, match="caller LLM failed"):
        await failing.next_utterance("Чем помочь?")

    silent = SimulatedCaller(ScriptedLLM(text(""), text("  ")), BY_ID["price_only"])
    with pytest.raises(CallerError, match="empty"):
        await silent.next_utterance("Чем помочь?")


# --- One call -------------------------------------------------------------------------------------


async def run(scenario, agent, caller, tmp_path, index=0):
    return await run_once(
        scenario,
        index,
        settings=make_settings(tmp_path),
        agent_llm=agent,
        caller_llm=caller,
        clock=lambda: NOW,
    )


async def test_a_whole_booking_call_produces_the_records_and_the_ordered_event_log(tmp_path):
    agent = ScriptedLLM(
        tool_round("prepare_booking", BOOKING_ARGS),  # turn 1: the model has everything
        tool_round("confirm_booking"),  # turn 2: the caller said yes
        [TextDelta("Всего доброго!"), *tool_round("end_call")],  # turn 3
    )
    caller = ScriptedLLM(
        text("Запишите меня на полировку завтра на 14:00, Игорь, 8 916 123 45 67, Камри."),
        text("Да, всё верно."),
        text("Нет, спасибо. До свидания. [КОНЕЦ]"),
    )

    result = await run(BY_ID["happy_path_booking"], agent, caller, tmp_path)

    assert result.outcome == "completed" and result.turns == 3
    assert result.caller_lines[1] == "Да, всё верно."
    assert [(i.turn, i.tool, i.committed) for i in result.items if i.kind == "tool"] == [
        (1, "prepare_booking", False),
        (2, "confirm_booking", True),
        (3, "end_call", False),
    ]
    assert result.ended_call() and any(i.kind == "end" and i.turn == 3 for i in result.items)
    [booking] = result.bookings
    assert (booking.phone, booking.preferred_time.hour, booking.caller_phone) == (
        "+79161234567",
        14,
        "+79991234567",
    )
    assert result.messages == []
    # the greeting is turn 0, and the read-back was spoken (by code) in turn 1
    assert result.items[0].turn == 0 and result.items[0].kind == "say"
    assert any(i.kind == "say" and i.turn == 1 and "Проверьте" in i.text for i in result.items)
    # the caller heard the read-back, not the tool calls:
    assert "Проверьте, пожалуйста" in caller.calls[1][-1].content
    assert "prepare_booking" not in " ".join(m.content for m in caller.calls[1])


async def test_the_call_ends_when_the_caller_is_done_even_if_the_agent_does_not_hang_up(tmp_path):
    agent = ScriptedLLM(text("Здравствуйте, слушаю вас."), text("Всего доброго."))
    caller = ScriptedLLM(text("Сколько стоит керамика?"), text("Спасибо. [КОНЕЦ]"))

    result = await run(BY_ID["price_only"], agent, caller, tmp_path)

    assert result.outcome == "completed" and result.turns == 2 and not result.ended_call()


async def test_the_caller_line_that_ends_the_call_is_still_delivered_to_the_agent(tmp_path):
    agent = ScriptedLLM(text("Ответ раз."), text("Всего доброго!"))
    caller = ScriptedLLM(text("Вопрос."), text("Пока. [КОНЕЦ]"))

    result = await run(BY_ID["price_only"], agent, caller, tmp_path)

    assert [m.content for m in agent.calls[1] if m.role is Role.USER][-1] == "Пока."
    assert "Всего доброго!" in result.speech()


async def test_hitting_the_turn_cap_is_inconclusive_not_a_failure(tmp_path):
    scenario = replace(BY_ID["price_only"], max_turns=3)
    agent = ScriptedLLM(*[text("Слушаю вас.")] * 3)
    caller = ScriptedLLM(*[text("Ну и?")] * 3)

    result = await run(scenario, agent, caller, tmp_path)

    assert result.outcome == "inconclusive" and result.turns == 3
    assert "no end after 3 turns" in result.detail


async def test_a_broken_caller_makes_the_run_inconclusive(tmp_path):
    result = await run(
        BY_ID["price_only"],
        ScriptedLLM(),
        ScriptedLLM(LLMError("down"), LLMError("down")),
        tmp_path,
    )
    assert (
        result.outcome == "inconclusive"
        and result.turns == 0
        and "caller LLM failed" in result.detail
    )


async def test_an_agent_side_llm_failure_is_an_infra_error_not_a_behaviour_failure(tmp_path):
    boom = LLMError("backend down")
    agent = ScriptedLLM(boom, boom)  # the failed round and its retry
    caller = ScriptedLLM(text("Сколько стоит?"), text("Алло? [КОНЕЦ]"))

    result = await run(BY_ID["price_only"], agent, caller, tmp_path)

    assert result.outcome == "infra_error" and "backend down" in result.detail
    assert result.turns == 1  # the run stopped at the failure: no tokens wasted on a void run
    assert any(i.kind == "failed" for i in result.items)


async def test_an_unexpected_exception_is_reported_not_raised(tmp_path):
    class Exploding(ScriptedLLM):
        async def stream(self, messages, tools=None):
            raise RuntimeError("bug in the agent")
            yield  # pragma: no cover

    result = await run(
        BY_ID["price_only"], Exploding(), ScriptedLLM(text("Сколько стоит?")), tmp_path
    )

    assert result.outcome == "infra_error" and "RuntimeError" in result.detail


async def test_records_saved_before_an_unexpected_exception_are_kept(tmp_path):
    agent = ScriptedLLM(
        tool_round("prepare_booking", BOOKING_ARGS),
        tool_round("confirm_booking"),
        RuntimeError("bug after the save"),
    )
    caller = ScriptedLLM(text("Запишите меня."), text("Да."), text("Спасибо."))

    result = await run(BY_ID["happy_path_booking"], agent, caller, tmp_path)

    assert result.outcome == "infra_error" and "RuntimeError" in result.detail
    assert len(result.bookings) == 1  # what had been saved is still reported


async def test_each_run_gets_its_own_database_and_the_clock_is_fixed(tmp_path):
    def booking_agent():
        return ScriptedLLM(
            tool_round("prepare_booking", BOOKING_ARGS), tool_round("confirm_booking")
        )

    caller_lines = [text("Запишите меня."), text("Да. [КОНЕЦ]")]
    first = await run(
        BY_ID["happy_path_booking"], booking_agent(), ScriptedLLM(*caller_lines), tmp_path, 0
    )
    second = await run(
        BY_ID["happy_path_booking"], booking_agent(), ScriptedLLM(*caller_lines), tmp_path, 1
    )

    assert first.outcome == second.outcome == "completed"
    assert len(first.bookings) == len(second.bookings) == 1  # not 2: nothing leaks between runs
    assert first.bookings[0].created_at == NOW


async def test_the_run_id_is_set_for_token_attribution(tmp_path):
    await run(
        BY_ID["price_only"],
        ScriptedLLM(text("Слушаю.")),
        ScriptedLLM(text("Пока [КОНЕЦ]")),
        tmp_path,
        7,
    )
    assert current_run.get() == "price_only#7"


# --- Cost -----------------------------------------------------------------------------------------


def test_usage_accumulates_and_prices_per_million_tokens():
    usage = Usage()
    usage.add({"prompt_tokens": 2_000_000, "completion_tokens": 500_000})
    usage.add({"prompt_tokens": 1_000_000})
    assert (usage.prompt, usage.completion, usage.requests) == (3_000_000, 500_000, 2)
    assert usage.cost((0.10, 0.40)) == pytest.approx(0.3 + 0.2)
    assert usage.cost(None) is None


def test_the_meter_attributes_usage_to_roles_and_to_the_running_scenario():
    meter = UsageMeter()
    token = current_run.set("price_only#2")
    try:
        meter.hook("agent")({"prompt_tokens": 2200, "completion_tokens": 40})
        meter.hook("caller")({"prompt_tokens": 300, "completion_tokens": 20})
    finally:
        current_run.reset(token)
    meter.hook("agent")({"prompt_tokens": 1, "completion_tokens": 1})  # outside any run

    assert meter.by_role["agent"].prompt == 2201 and meter.by_role["caller"].completion == 20
    assert meter.by_scenario["price_only"]["agent"].prompt == 2200
    assert meter.by_scenario["price_only"]["caller"].prompt == 300
    assert list(meter.by_scenario) == ["price_only"]


def test_total_cost_needs_a_price_for_every_role():
    meter = UsageMeter()
    meter.hook("agent")({"prompt_tokens": 1_000_000, "completion_tokens": 0})
    assert meter.total_cost({"agent": (0.15, 0.6)}) is None
    assert meter.total_cost({"agent": (0.15, 0.6), "caller": (0.1, 0.4)}) == pytest.approx(0.15)


def test_estimate_scales_with_the_number_of_runs_and_needs_both_prices():
    prices = {"agent": (0.15, 0.6), "caller": (0.1, 0.4)}
    one = estimate_cost(1, prices)
    assert one == pytest.approx(0.15 * 0.025 + 0.6 * 0.0007 + 0.1 * 0.009 + 0.4 * 0.0005)
    assert estimate_cost(50, prices) == pytest.approx(50 * one)
    assert estimate_cost(50, {"agent": (0.15, 0.6)}) is None


async def test_prices_come_from_the_openrouter_model_list():
    listing = {
        "data": [
            {"id": "a/model", "pricing": {"prompt": "0.00000015", "completion": "0.0000006"}},
            {"id": "b/model", "pricing": {"prompt": "0.0000001", "completion": "0.0000004"}},
            {"id": "c/other", "pricing": {"prompt": "1", "completion": "1"}},
            {"id": "d/broken", "pricing": {}},
        ]
    }
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=listing))
    )

    prices = await fetch_prices(
        "https://openrouter.ai/api/v1", ["a/model", "b/model", "d/broken", "missing/x"], client
    )

    assert prices == {
        "a/model": (pytest.approx(0.15), pytest.approx(0.6)),
        "b/model": (pytest.approx(0.1), pytest.approx(0.4)),
    }


async def test_no_prices_for_other_backends_or_on_failure():
    assert await fetch_prices("http://localhost:11434/v1", ["qwen3.5:4b"]) == {}
    failing = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert await fetch_prices("https://openrouter.ai/api/v1", ["a/b"], failing) == {}


# --- Report ---------------------------------------------------------------------------------------


def graded(scenario_id, status, *, failed=(), passed_checks=("a",), turns=4, index=0) -> Graded:
    outcome = "completed" if status in ("pass", "fail") else status
    run = RunResult(
        scenario_id, index, ["x"] * turns, [Item(1, "say", text="ok")], [], [], outcome, "why"
    )
    results = [CheckResult(n, True) for n in passed_checks]
    results += [CheckResult(n, False, f"detail of {n}") for n in failed]
    return Graded(run, results)


def test_status_of_a_graded_run():
    assert graded("s", "pass").status == "pass"
    assert graded("s", "fail", failed=["x"]).status == "fail"
    assert graded("s", "infra_error", failed=["x"]).status == "infra_error"
    assert graded("s", "inconclusive").status == "inconclusive"


def test_pass_rates_exclude_infra_errors_and_inconclusive_runs():
    runs = [
        graded("one", "pass"),
        graded("one", "pass"),
        graded("one", "fail", failed=["x"]),
        graded("one", "infra_error"),
        graded("one", "inconclusive"),
        graded("two", "infra_error"),
    ]

    summary = summarize(runs, ["one", "two"])

    one, two = summary["one"], summary["two"]
    assert (one.runs, one.passed, one.failed, one.infra, one.inconclusive) == (5, 2, 1, 1, 1)
    assert one.gradable == 3 and one.pass_rate == pytest.approx(2 / 3)
    assert two.pass_rate is None and two.gradable == 0


def test_per_check_counts_use_gradable_runs_only():
    runs = [
        graded("s", "pass", passed_checks=("a", "b")),
        graded("s", "fail", passed_checks=("a",), failed=["b"]),
        graded("s", "infra_error", passed_checks=("a",), failed=["b"]),  # ignored
    ]
    s = summarize(runs, ["s"])["s"]
    assert s.checks == {"a": [2, 2], "b": [1, 2]}
    assert s.examples == {"b": "detail of b"}


def test_the_report_shows_rates_failing_checks_and_the_invariant_matrix():
    invariants = tuple(INVARIANT_NAMES)
    runs = [
        graded("alpha", "pass", passed_checks=invariants + ("time_ok",)),
        graded(
            "alpha", "fail", passed_checks=invariants[1:] + ("time_ok",), failed=[invariants[0]]
        ),
        graded("beta", "fail", passed_checks=invariants, failed=["date_ok"]),
        graded("beta", "infra_error"),
    ]

    report = format_report(runs, ["alpha", "beta"])

    assert "PASS RATE PER SCENARIO" in report and "PER-CHECK BREAKDOWN" in report
    alpha = next(line for line in report.splitlines() if line.startswith("alpha "))
    assert "50%" in alpha
    beta = next(line for line in report.splitlines() if line.startswith("beta "))
    assert "0%" in beta and "   1" in beta  # one infra error is shown, not counted
    assert "✗ date_ok" in report and "detail of date_ok" in report
    assert f"✗ {invariants[0]}" in report
    assert all(name in report for name in invariants)
    assert "1/2" in report  # the failed invariant in `alpha`


def test_the_transcript_shows_both_sides_the_tools_the_records_and_failed_checks():
    run = RunResult(
        "s",
        3,
        ["Хочу записаться", "Да"],
        [
            Item(0, "say", text="Здравствуйте!"),
            Item(1, "say", text="Как вас зовут?"),
            Item(2, "tool", tool="confirm_booking", args={"a": 1}, text="принято", committed=True),
            Item(2, "end"),
        ],
        [],
        [],
        "completed",
    )
    text_ = format_transcript(Graded(run, [CheckResult("time_ok", False, "wrong time")]))

    for expected in (
        "s #3: FAIL",
        "КЛИЕНТ: Хочу записаться",
        "АГЕНТ: Как вас зовут?",
        "[tool] confirm_booking",
        "SAVED принято",
        "[конец звонка]",
        "✗ time_ok: wrong time",
    ):
        assert expected in text_


def test_the_json_export_is_serialisable_and_complete():
    dump = to_json([graded("s", "fail", failed=["x"])])
    round_tripped = json.loads(json.dumps(dump, ensure_ascii=False))
    assert round_tripped[0]["status"] == "fail" and round_tripped[0]["scenario"] == "s"
    assert {c["name"] for c in round_tripped[0]["checks"]} == {"a", "x"}


# --- The runner and the command line --------------------------------------------------------------


def probe_client(outcomes):
    """A client for pick_caller_model: {model: (status, content)}; records what was asked."""
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        asked.append(model)
        status, content = outcomes.get(model, (404, None))
        body = {"choices": [{"message": {"content": content}}]} if status == 200 else {"error": {}}
        return httpx.Response(status, json=body)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x/v1")
    return client, asked


async def test_the_caller_model_is_the_first_working_candidate_from_another_family(tmp_path):
    client, asked = probe_client(
        {"google/a": (404, None), "openai/b": (200, "да"), "meta/c": (200, "да")}
    )

    model, note = await evals_main.pick_caller_model(
        make_settings(tmp_path), ("deepseek/skipped", "google/a", "openai/b", "meta/c"), client
    )

    assert model == "openai/b"
    assert asked == ["google/a", "openai/b"]  # same family skipped, later candidates untouched
    assert "google/a: HTTP 404" in note


async def test_the_agents_own_model_is_only_a_last_resort(tmp_path):
    client, _ = probe_client({})
    settings = make_settings(tmp_path)

    model, note = await evals_main.pick_caller_model(settings, ("google/a", "openai/b"), client)

    assert model == settings.llm_model and note.startswith("FALLBACK")


async def test_an_empty_answer_does_not_count_as_a_working_candidate(tmp_path):
    client, _ = probe_client({"google/a": (200, ""), "openai/b": (200, "да")})
    model, _ = await evals_main.pick_caller_model(
        make_settings(tmp_path), ("google/a", "openai/b"), client
    )
    assert model == "openai/b"


def test_the_default_caller_candidates_are_not_from_the_agents_family():
    assert all(not m.startswith("deepseek/") for m in evals_main.CALLER_MODEL_CANDIDATES)
    assert len(evals_main.CALLER_MODEL_CANDIDATES) >= 3


def test_defaults_match_the_agreed_ones():
    args = evals_main.build_parser().parse_args([])
    assert (args.runs, args.concurrency, args.max_cost) == (5, 4, 2.0)
    assert args.dry_run is False and args.show_failures is False and args.min_pass_rate is None


def test_select_scenarios():
    assert [s.id for s in evals_main.select_scenarios(None)] == [s.id for s in SCENARIOS]
    assert [s.id for s in evals_main.select_scenarios("price_only, sunday_closed")] == [
        "price_only",
        "sunday_closed",
    ]
    with pytest.raises(SystemExit, match="unknown scenario 'nope'"):
        evals_main.select_scenarios("price_only,nope")


def test_the_fixed_clock_is_a_friday_afternoon_in_the_business_zone(tmp_path):
    now = evals_main.fixed_now(make_settings(tmp_path))
    assert now.strftime("%A %Y-%m-%d %H:%M") == "Friday 2026-09-25 15:00"
    assert now.tzinfo.key == "Europe/Moscow"


def test_list_prints_every_scenario(capsys):
    assert evals_main.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert all(s.id in out for s in SCENARIOS)


@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    settings = make_settings(tmp_path)
    monkeypatch.setattr(evals_main, "get_settings", lambda: settings)

    async def fake_prices(base_url, models):
        return {m: (0.15, 0.6) for m in models}

    monkeypatch.setattr(evals_main, "fetch_prices", fake_prices)
    return settings


def test_dry_run_prints_the_plan_and_the_estimate_and_makes_no_llm_calls(cli_env, capsys):
    assert evals_main.main(["--dry-run", "--runs", "3", "--scenarios", "price_only"]) == 0

    out = capsys.readouterr().out
    assert "agent under test : deepseek/deepseek-v4.1-flash" in out
    assert "simulated caller : google/gemini-2.5-flash-lite" in out and "would be probed" in out
    assert "1 scenarios x 3 runs = 3 calls" in out
    assert "estimated cost   : about $" in out


def test_a_sweep_estimated_above_max_cost_is_refused_before_any_llm_call(cli_env, capsys):
    status = evals_main.main(["--caller-model", "openai/x", "--max-cost", "0.001"])

    assert status == 2
    assert "EXCEEDS --max-cost" in capsys.readouterr().out


def test_bad_numbers_are_rejected(cli_env, capsys):
    assert evals_main.main(["--runs", "0"]) == 2


async def test_a_sweep_runs_scenarios_round_robin_and_grades_each_run(tmp_path):
    settings = make_settings(tmp_path)
    scenarios = [BY_ID["price_only"], BY_ID["address_only"]]

    class Agent(ScriptedLLM):
        async def stream(self, messages, tools=None):
            self.calls.append(list(messages))
            for event in text(
                "Керамика — от двадцати пяти тысяч рублей. Точную стоимость определит мастер."
            ):
                yield event

    class Caller(ScriptedLLM):
        async def stream(self, messages, tools=None):
            self.calls.append(list(messages))
            for event in text("Понятно, спасибо. [КОНЕЦ]"):
                yield event

    results = await evals_main.sweep(
        scenarios,
        2,
        settings=settings,
        agent_llm=Agent(),
        caller_llm=Caller(),
        concurrency=2,
        stop_check=lambda: False,
    )

    assert len(results) == 4
    assert {(g.run.scenario_id, g.run.run_index) for g in results} == {
        ("price_only", 0),
        ("address_only", 0),
        ("price_only", 1),
        ("address_only", 1),
    }
    assert all(g.run.outcome == "completed" and g.results for g in results)
    price_runs = [g for g in results if g.run.scenario_id == "price_only"]
    assert all(g.status == "pass" for g in price_runs)  # this canned agent answers price_only right


async def test_a_sweep_stops_starting_new_runs_when_told_to(tmp_path):
    started = []

    class Agent(ScriptedLLM):
        async def stream(self, messages, tools=None):
            started.append(1)
            for event in text("Слушаю вас."):
                yield event

    class Caller(ScriptedLLM):
        async def stream(self, messages, tools=None):
            for event in text("Пока. [КОНЕЦ]"):
                yield event

    results = await evals_main.sweep(
        [BY_ID["price_only"]],
        6,
        settings=make_settings(tmp_path),
        agent_llm=Agent(),
        caller_llm=Caller(),
        concurrency=1,
        stop_check=lambda: len(started) >= 2,
    )

    assert 1 <= len(results) < 6


async def test_sweep_grading_uses_the_scenario_checks_and_the_invariants(tmp_path):
    class Agent(ScriptedLLM):
        async def stream(self, messages, tools=None):
            for event in text("Записал вас, готово."):  # a forbidden word, no records
                yield event

    class Caller(ScriptedLLM):
        async def stream(self, messages, tools=None):
            for event in text("Спасибо. [КОНЕЦ]"):
                yield event

    [result] = await evals_main.sweep(
        [BY_ID["price_only"]],
        1,
        settings=make_settings(tmp_path),
        agent_llm=Agent(),
        caller_llm=Caller(),
        concurrency=1,
        stop_check=lambda: False,
    )

    failed = {r.name for r in result.results if not r.passed}
    assert "no_записал_before_confirm" in failed and "quotes_25000" in failed
    assert result.status == "fail"
    names = {r.name for r in result.results}
    assert set(INVARIANT_NAMES) <= names and "quotes_25000" in names
