"""Offline tests for the CLI: argument handling, the conversation loop, output, exit paths."""

import asyncio
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from agent import cli
from agent.cli import (
    Options,
    ScriptReader,
    StdinReader,
    UsageError,
    build_parser,
    parse_options,
    run,
)
from agent.llm import LLMError, StreamEnd, TextDelta, ToolCall, ToolCallEvent
from agent.records import CallbackMessage
from agent.settings import Settings
from agent.storage import SqliteSink

MOSCOW = ZoneInfo("Europe/Moscow")
REPO_CONFIG = Path(__file__).parent.parent / "config" / "business.yaml"


def make_settings(tmp_path, **overrides) -> Settings:
    values = {
        "llm_base_url": "http://llm.test/v1",
        "llm_api_key": "key",
        "llm_model": "test-model",
        "telegram_bot_token": "123:abc",
        "telegram_chat_id": "1",
        "business_config_path": REPO_CONFIG,
        "db_path": tmp_path / "agent.db",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


class ScriptedLLM:
    def __init__(self, *scripts):
        self._scripts = list(scripts)
        self.calls = []

    async def stream(self, messages, tools=None):
        self.calls.append(list(messages))
        script = self._scripts.pop(0)
        if isinstance(script, Exception):
            raise script
        for item in script:
            yield item


class FakeNotifier:
    def __init__(self, error=None):
        self.sent = []
        self._error = error

    async def send(self, text):
        if self._error:
            raise self._error
        self.sent.append(text)


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
    "preferred_period": "день",
    "notes": "после обеда",
}


def options(**overrides) -> Options:
    values = {
        "caller": None,
        "notify": False,
        "db": None,
        "now": datetime(2026, 9, 26, 15, 0, tzinfo=MOSCOW),
        "script": None,
        "show_tools": False,
    }
    values.update(overrides)
    return Options(**values)


class Lines:
    """A scripted caller: hands out lines, then EOF (None)."""

    def __init__(self, *lines):
        self._lines = list(lines)
        self.prompts = []

    async def __call__(self, prompt):
        self.prompts.append(prompt)
        return self._lines.pop(0) if self._lines else None


async def call_cli(tmp_path, lines, *scripts, opts=None, settings=None, **kwargs):
    out: list[str] = []
    llm = ScriptedLLM(*scripts)
    status = await run(
        opts or options(),
        settings or make_settings(tmp_path),
        Lines(*lines),
        out.append,
        llm=llm,
        **kwargs,
    )
    assert status == 0
    return out, llm


# --- Arguments ------------------------------------------------------------------------------------


def parse(*argv, tmp_path=None):
    settings = Settings(
        _env_file=None,
        llm_base_url="http://x/v1",
        llm_api_key="k",
        llm_model="m",
        telegram_bot_token="1:a",
        telegram_chat_id="1",
    )
    return parse_options(build_parser().parse_args(list(argv)), settings)


def test_defaults_are_safe_notifications_off_no_fake_time_no_script():
    opts = parse()
    assert opts == Options(None, False, None, None, None, False)


def test_caller_is_normalized():
    assert parse("--caller", "8 (916) 123-45-67").caller == "+79161234567"


def test_a_bad_caller_is_a_usage_error():
    with pytest.raises(UsageError, match="not a Russian phone number"):
        parse("--caller", "12345")


def test_now_without_a_zone_gets_the_business_time_zone():
    assert parse("--now", "2026-09-26 15:00").now == datetime(2026, 9, 26, 15, 0, tzinfo=MOSCOW)


def test_now_with_an_explicit_offset_is_kept():
    now = parse("--now", "2026-09-26T15:00:00+05:00").now
    assert now.utcoffset().total_seconds() == 5 * 3600


def test_a_bad_now_is_a_usage_error():
    with pytest.raises(UsageError, match="expected 'YYYY-MM-DD HH:MM'"):
        parse("--now", "завтра")


def test_script_lines_skip_blanks_and_comments(tmp_path):
    script = tmp_path / "s.txt"
    script.write_text("# a comment\n\nПривет\n   \n  Хочу записаться  \n", encoding="utf-8")

    assert parse("--script", str(script)).script == ["Привет", "Хочу записаться"]


def test_a_missing_script_file_is_a_usage_error(tmp_path):
    with pytest.raises(UsageError, match="cannot read --script"):
        parse("--script", str(tmp_path / "nope.txt"))


def test_flags_are_passed_through(tmp_path):
    opts = parse("--notify", "--show-tools", "--db", str(tmp_path / "x.db"))
    assert opts.notify and opts.show_tools and opts.db == tmp_path / "x.db"


# --- The conversation -----------------------------------------------------------------------------


async def test_greeting_first_then_the_dialogue_then_hang_up_on_eof(tmp_path):
    out, llm = await call_cli(tmp_path, ["Сколько стоит мойка?"], text("От трёх тысяч рублей."))

    lines = [line for line in out if line]
    assert lines[2] == "АГЕНТ: Здравствуйте!"
    assert "АГЕНТ: Чем могу помочь?" in out
    assert "АГЕНТ: От трёх тысяч рублей." in out
    assert "(the caller hung up)" in "\n".join(out)
    assert "(nothing)" in out  # nothing was saved


async def test_header_shows_model_time_caller_and_notification_state(tmp_path):
    out, _ = await call_cli(tmp_path, [], opts=options(caller="+79991234567"))

    assert "model: test-model   time: Saturday 2026-09-26 15:00" in out[0]
    assert "caller: +79991234567   notifications: off" in out[1]


async def test_hidden_number_is_shown_as_such(tmp_path):
    out, _ = await call_cli(tmp_path, [])
    assert "caller: hidden number" in out[1]


async def test_the_model_sees_the_greeting_and_the_faked_time(tmp_path):
    _, llm = await call_cli(tmp_path, ["Привет"], text("Слушаю вас."))

    system, greeting, user = llm.calls[0]
    assert "Сейчас суббота, 26 сентября 2026, 15:00" in system.content
    assert "Номер звонящего: не определён." in system.content
    assert greeting.content.startswith("Здравствуйте!") and user.content == "Привет"


async def test_the_caller_id_reaches_the_prompt(tmp_path):
    _, llm = await call_cli(
        tmp_path, ["Привет"], text("Слушаю вас."), opts=options(caller="+79991234567")
    )
    assert "Номер звонящего: +79991234567." in llm.calls[0][0].content


async def test_quit_hangs_up_and_blank_lines_are_ignored(tmp_path):
    out, llm = await call_cli(tmp_path, ["", "   ", "/quit", "не должно быть прочитано"])

    assert llm.calls == []
    assert "(the caller hung up)" in out


async def test_script_mode_echoes_the_callers_lines_and_says_the_script_ended(tmp_path):
    out, _ = await call_cli(
        tmp_path, ["Привет"], text("Слушаю вас."), opts=options(script=["Привет"])
    )

    assert "КЛИЕНТ: Привет" in out
    assert "(script ended, the caller hangs up)" in out


async def test_interactive_mode_does_not_echo_because_the_terminal_already_did(tmp_path):
    out, _ = await call_cli(tmp_path, ["Привет"], text("Слушаю вас."))
    assert "КЛИЕНТ: Привет" not in out


async def test_the_reader_is_prompted_for_each_line(tmp_path):
    reader = Lines("раз", "два")
    out: list[str] = []
    await run(
        options(),
        make_settings(tmp_path),
        reader,
        out.append,
        llm=ScriptedLLM(text("Ответ раз."), text("Ответ два.")),
    )
    assert reader.prompts == ["КЛИЕНТ: "] * 3  # two lines and the EOF


# --- A booking through the CLI --------------------------------------------------------------------


async def test_full_booking_reads_back_saves_and_shows_the_record(tmp_path):
    out, _ = await call_cli(
        tmp_path,
        ["Меня зовут Игорь, номер 8 916 123 45 67", "Да, всё верно", "Нет, спасибо"],
        tool_round("prepare_booking", BOOKING_ARGS),
        tool_round("confirm_booking"),
        tool_round("end_call"),
        opts=options(caller="+79991234567"),
    )

    joined = "\n".join(out)
    assert "АГЕНТ: Проверьте, пожалуйста: Игорь, полировка кузова" in joined
    assert "Номер телефона заканчивается на четыре пять шесть семь." in joined
    assert "АГЕНТ: Всё верно?" in out
    assert (
        "АГЕНТ: Заявка принята и передана администратору, он перезвонит для подтверждения." in out
    )
    assert "АГЕНТ: Могу ещё чем-то помочь?" in out  # spoken by code, no LLM round needed
    assert "[конец звонка] агент положил трубку" in joined
    assert "=== Saved during this call ===" in joined
    assert "--- Telegram: не отправлено ---" in joined  # notifications are off
    assert "Новая заявка №1" in joined and "Комментарий: после обеда" in joined

    store = await SqliteSink.open(tmp_path / "agent.db")
    try:
        [stored] = await store.list_bookings()
        assert stored.booking.caller_phone == "+79991234567"
    finally:
        await store.close()


async def test_show_tools_prints_calls_results_and_timings(tmp_path):
    out, _ = await call_cli(
        tmp_path,
        ["Меня зовут Игорь, номер 8 916 123 45 67"],
        tool_round("prepare_booking", BOOKING_ARGS),
        opts=options(show_tools=True),
    )

    joined = "\n".join(out)
    assert '[tool] prepare_booking({"name": "Игорь"' in joined
    assert "[result] Заявка подготовлена" in joined
    assert any(line.startswith("  (") and line.endswith("s)") for line in out)


async def test_tool_details_are_hidden_without_show_tools(tmp_path):
    out, _ = await call_cli(
        tmp_path,
        ["Меня зовут Игорь, номер 8 916 123 45 67"],
        tool_round("prepare_booking", BOOKING_ARGS),
    )

    joined = "\n".join(out)
    assert "[tool]" not in joined and "[result]" not in joined
    assert "АГЕНТ: Проверьте, пожалуйста" in joined  # the caller still hears the read-back


async def test_db_command_lists_every_saved_record_not_just_this_calls(tmp_path):
    settings = make_settings(tmp_path)
    store = await SqliteSink.open(settings.db_path)
    await store.add_message(
        CallbackMessage(
            "вчерашняя просьба", "Анна", None, None, datetime(2026, 9, 25, 10, 0, tzinfo=MOSCOW)
        )
    )
    await store.close()

    out, _ = await call_cli(tmp_path, ["/db", "/quit"], settings=settings)

    joined = "\n".join(out)
    assert "=== All saved records ===" in joined and "вчерашняя просьба" in joined
    # ...while the end-of-call summary only shows this call's records:
    assert joined.count("вчерашняя просьба") == 1
    assert "=== Saved during this call ===" in joined


# --- LLM failures ---------------------------------------------------------------------------------


async def test_two_failed_turns_show_the_failures_the_apology_and_the_saved_callback(tmp_path):
    boom = LLMError("backend down")
    out, _ = await call_cli(
        tmp_path,
        ["Хочу записаться", "Алло?"],
        boom,
        boom,
        boom,
        boom,
        opts=options(caller="+79991234567"),
    )

    joined = "\n".join(out)
    assert "[сбой] backend down (подряд: 1)" in joined
    assert "АГЕНТ: Простите, плохо слышно. Повторите, пожалуйста." in out
    assert "[сбой] backend down (подряд: 2)" in joined
    assert "АГЕНТ: Извините, у нас возникли технические неполадки." in out
    assert "[конец звонка] агент положил трубку" in joined
    assert "Сообщение для администратора №1" in joined
    assert "«Хочу записаться»; «Алло?»" in joined and "Звонил с номера: +79991234567" in joined


# --- Notifications --------------------------------------------------------------------------------


async def test_notify_off_never_touches_the_notifier(tmp_path):
    notifier = FakeNotifier()
    await call_cli(
        tmp_path,
        ["Меня зовут Игорь, номер 8 916 123 45 67", "Да"],
        tool_round("prepare_booking", BOOKING_ARGS),
        tool_round("confirm_booking"),
        text("Готово."),
        notifier=notifier,
    )
    assert notifier.sent == []


async def test_notify_delivers_and_waits_for_delivery_before_exiting(tmp_path):
    notifier = FakeNotifier()
    out, _ = await call_cli(
        tmp_path,
        ["Меня зовут Игорь, номер 8 916 123 45 67", "Да"],
        tool_round("prepare_booking", BOOKING_ARGS),
        tool_round("confirm_booking"),
        text("Готово."),
        opts=options(notify=True),
        notifier=notifier,
    )

    assert len(notifier.sent) == 1 and notifier.sent[0].startswith("Новая заявка №1")
    assert "notifications: ON (real Telegram)" in out[1]
    store = await SqliteSink.open(tmp_path / "agent.db")
    try:
        assert await store.list_unnotified() == []  # marked before the process would exit
    finally:
        await store.close()


async def test_notify_warns_about_and_resends_older_unnotified_records(tmp_path):
    settings = make_settings(tmp_path)
    store = await SqliteSink.open(settings.db_path)
    await store.add_message(
        CallbackMessage("старая просьба", None, None, None, datetime(2026, 9, 1, tzinfo=MOSCOW))
    )
    await store.close()
    notifier = FakeNotifier()

    out, _ = await call_cli(
        tmp_path, [], opts=options(notify=True), settings=settings, notifier=notifier
    )

    assert "(resending 1 older unnotified record(s) to Telegram)" in out
    assert "старая просьба" in notifier.sent[0]


async def test_undeliverable_notifications_are_reported_and_stay_in_the_database(tmp_path):
    from agent.notifier import NotifyError

    notifier = FakeNotifier(
        error=NotifyError("Telegram rejected the message (403)", permanent=True)
    )
    out, _ = await call_cli(
        tmp_path,
        ["Меня зовут Игорь, номер 8 916 123 45 67", "Да"],
        tool_round("prepare_booking", BOOKING_ARGS),
        tool_round("confirm_booking"),
        text("Готово."),
        opts=options(notify=True),
        notifier=notifier,
        delivery_wait=0.3,
    )

    assert any("not all notifications were delivered" in line for line in out)
    store = await SqliteSink.open(tmp_path / "agent.db")
    try:
        assert len(await store.list_unnotified()) == 1
    finally:
        await store.close()


# --- Readers --------------------------------------------------------------------------------------


async def test_script_reader_returns_lines_then_none():
    reader = ScriptReader(["раз", "два"])
    assert [await reader("p"), await reader("p"), await reader("p")] == ["раз", "два", None]


async def test_stdin_reader_returns_each_line_then_none_and_prints_the_prompt(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("первая строка\nвторая\n"))
    reader = StdinReader()

    got = [await asyncio.wait_for(reader("КЛИЕНТ: "), 2) for _ in range(3)]

    assert got == ["первая строка", "вторая", None]
    assert capsys.readouterr().out == "КЛИЕНТ: " * 3


# --- main() ---------------------------------------------------------------------------------------


def test_main_reports_a_bad_configuration_without_printing_secret_values(monkeypatch, capsys):
    def broken_settings():
        monkeypatch.setenv("LLM_API_KEY", "sk-very-secret-value")
        for name in ("LLM_BASE_URL", "LLM_MODEL", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
            monkeypatch.delenv(name, raising=False)
        return Settings(_env_file=None)

    monkeypatch.setattr(cli, "get_settings", broken_settings)

    assert cli.main([]) == 1

    err = capsys.readouterr().err
    assert "configuration error" in err and "llm_base_url" in err
    assert "sk-very-secret-value" not in err


def test_main_exits_with_status_2_on_a_bad_argument(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "get_settings", lambda: make_settings(tmp_path))

    with pytest.raises(SystemExit) as info:
        cli.main(["--caller", "nonsense"])

    assert info.value.code == 2
    assert "not a Russian phone number" in capsys.readouterr().err


def test_main_runs_a_scripted_call_end_to_end(monkeypatch, tmp_path, capsys):
    script = tmp_path / "call.txt"
    script.write_text("Привет\n", encoding="utf-8")
    settings = make_settings(tmp_path)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    llm = ScriptedLLM(text("Слушаю вас."))
    real_run = cli.run

    async def run_with_fake_llm(opts, sett, reader, out=print, **kwargs):
        return await real_run(opts, sett, reader, out, llm=llm, **kwargs)

    monkeypatch.setattr(cli, "run", run_with_fake_llm)

    assert cli.main(["--script", str(script), "--now", "2026-09-26 15:00"]) == 0

    out = capsys.readouterr().out
    assert "АГЕНТ: Слушаю вас." in out and "КЛИЕНТ: Привет" in out


def test_main_reports_an_unusable_database_path_with_exit_status_1(monkeypatch, tmp_path, capsys):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    script = tmp_path / "s.txt"
    script.write_text("Привет\n", encoding="utf-8")
    monkeypatch.setattr(cli, "get_settings", lambda: make_settings(tmp_path))

    assert cli.main(["--db", str(blocker / "agent.db"), "--script", str(script)]) == 1
    assert "cannot open database" in capsys.readouterr().err


def test_parser_accepts_every_documented_option(tmp_path):
    namespace = build_parser().parse_args(
        [
            "--caller",
            "+79991234567",
            "--notify",
            "--db",
            "x.db",
            "--now",
            "2026-09-26 15:00",
            "--script",
            "s.txt",
            "--show-tools",
        ]
    )
    assert namespace.notify and namespace.show_tools and namespace.caller == "+79991234567"


async def test_the_cli_shows_what_the_speech_guard_blocked_and_speaks_the_rest(tmp_path):
    out, _ = await call_cli(tmp_path, ["Хочу записаться"], text("Хорошо, записал. Как вас зовут?"))

    assert "  [guard] blocked (written_down): Хорошо, записал." in out
    assert "АГЕНТ: Как вас зовут?" in out
    assert "АГЕНТ: Хорошо, записал." not in out
