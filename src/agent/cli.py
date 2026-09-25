"""Text-mode phone call in the terminal: the same conversation the phone line will have.

    .venv/bin/python -m agent.cli                      # interactive, no caller ID
    .venv/bin/python -m agent.cli --caller +79991234567 --show-tools
    .venv/bin/python -m agent.cli --script scenario.txt --now "2026-09-26 15:00"

Type as the caller. `/quit` (or Ctrl-D / Ctrl-C) hangs up; `/db` shows what is saved.
Nothing is sent to Telegram unless --notify is given. Records are saved to the database
(data/agent.db unless --db); note that --notify also resends every older record that is still
unnotified in that database.
"""

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from agent.app import Runtime, offset_clock, open_runtime
from agent.business import BusinessConfigError
from agent.dialogue import EndCall, Say, ToolResult
from agent.llm import LLMClient
from agent.notifier import Notifier, format_record
from agent.session import TurnFailed
from agent.settings import Settings, get_settings
from agent.storage import StorageError, StoredBooking, StoredMessage
from agent.tools import normalize_phone

ReadLine = Callable[[str], Awaitable[str | None]]  # prompt -> the caller's line, None = hung up
Out = Callable[[str], None]

DELIVERY_WAIT_SECONDS = 30.0  # how long --notify waits for Telegram before exiting


class UsageError(Exception):
    """A bad command-line value; main() prints it and exits with status 2."""


@dataclass(frozen=True)
class Options:
    caller: str | None
    notify: bool
    db: Path | None
    now: datetime | None
    script: list[str] | None  # the caller's lines, if running non-interactively
    show_tools: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent.cli", description="Talk to the booking agent in the terminal."
    )
    parser.add_argument(
        "--caller", metavar="PHONE", help="caller ID (default: none, a hidden number)"
    )
    parser.add_argument(
        "--notify", action="store_true", help="really send Telegram notifications (default: off)"
    )
    parser.add_argument("--db", type=Path, metavar="PATH", help="database file (default: DB_PATH)")
    parser.add_argument(
        "--now", metavar="'YYYY-MM-DD HH:MM'", help="pretend it is this time (business time zone)"
    )
    parser.add_argument(
        "--script",
        type=Path,
        metavar="FILE",
        help="read the caller's lines from FILE, one per line",
    )
    parser.add_argument(
        "--show-tools", action="store_true", help="print tool calls, their results and timings"
    )
    return parser


def parse_options(args: argparse.Namespace, settings: Settings) -> Options:
    caller = None
    if args.caller is not None:
        caller = normalize_phone(args.caller)
        if caller is None:
            raise UsageError(f"--caller {args.caller!r} is not a Russian phone number")

    now = None
    if args.now is not None:
        try:
            now = datetime.fromisoformat(args.now)
        except ValueError:
            raise UsageError(f"--now {args.now!r}: expected 'YYYY-MM-DD HH:MM'") from None
        if now.tzinfo is None:
            now = now.replace(tzinfo=settings.tz)

    script = None
    if args.script is not None:
        try:
            raw = args.script.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise UsageError(f"cannot read --script {args.script}: {exc.strerror}") from None
        script = [ln.strip() for ln in raw if ln.strip() and not ln.lstrip().startswith("#")]

    return Options(caller, args.notify, args.db, now, script, args.show_tools)


# --- Reading the caller ---------------------------------------------------------------------------


class ScriptReader:
    """Feeds the lines of a script file to the conversation, then hangs up."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = list(lines)

    async def __call__(self, prompt: str) -> str | None:
        return self._lines.pop(0) if self._lines else None


class StdinReader:
    """Reads stdin on a daemon thread, so a blocked read never keeps the process alive."""

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _push(self, item: str | None) -> None:
        with contextlib.suppress(RuntimeError):  # the loop is already closed: we are exiting
            self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    def _pump(self) -> None:
        for line in sys.stdin:
            self._push(line.rstrip("\n"))
        self._push(None)

    async def __call__(self, prompt: str) -> str | None:
        print(prompt, end="", flush=True)
        return await self._queue.get()


# --- The call -------------------------------------------------------------------------------------


async def run(
    options: Options,
    settings: Settings,
    read_line: ReadLine,
    out: Out = print,
    *,
    llm: LLMClient | None = None,
    notifier: Notifier | None = None,
    delivery_wait: float = DELIVERY_WAIT_SECONDS,
) -> int:
    clock = offset_clock(options.now, lambda: datetime.now(settings.tz)) if options.now else None
    async with open_runtime(
        settings,
        notify=options.notify,
        db_path=options.db,
        clock=clock,
        llm=llm,
        notifier=notifier,
    ) as runtime:
        session = runtime.new_call(options.caller)
        before = (
            len(await runtime.store.list_bookings()),
            len(await runtime.store.list_messages()),
        )

        out(f"model: {settings.llm_model}   time: {runtime.clock():%A %Y-%m-%d %H:%M}")
        out(
            f"caller: {options.caller or 'hidden number'}   notifications: "
            f"{'ON (real Telegram)' if options.notify else 'off'}"
        )
        if runtime.resent_at_start:
            out(f"(resending {runtime.resent_at_start} older unnotified record(s) to Telegram)")
        out("")

        for say in session.greet():
            out(f"АГЕНТ: {say.text}")

        scripted = options.script is not None
        while not session.ended:
            line = await read_line("КЛИЕНТ: ")
            if line is None:
                out("(script ended, the caller hangs up)" if scripted else "\n(the caller hung up)")
                break
            line = line.strip()
            if not line:
                continue
            if line in ("/quit", "/exit"):
                out("(the caller hung up)")
                break
            if line == "/db":
                await _print_records(runtime, out)
                continue
            if scripted:
                out(f"КЛИЕНТ: {line}")
            await _handle_turn(session, line, options.show_tools, out)

        out("")
        await _print_records(runtime, out, since=before)
        if options.notify:
            await _wait_for_delivery(runtime, delivery_wait, out)
    return 0


async def _handle_turn(session, line: str, show_tools: bool, out: Out) -> None:
    started = time.monotonic()
    async for event in session.handle(line):
        if isinstance(event, Say):
            out(f"АГЕНТ: {event.text}")
        elif isinstance(event, ToolResult):
            if show_tools:
                args = json.loads(event.call.arguments or "{}")
                out(f"  [tool] {event.call.name}({json.dumps(args, ensure_ascii=False)})")
                out(f"  [result] {event.result}")
        elif isinstance(event, TurnFailed):
            out(f"  [сбой] {event.reason} (подряд: {event.consecutive})")
        elif isinstance(event, EndCall):
            out("  [конец звонка] агент положил трубку")
    if show_tools:
        out(f"  ({time.monotonic() - started:.1f}s)")


async def _print_records(
    runtime: Runtime, out: Out, *, since: tuple[int, int] | None = None
) -> None:
    """Print saved records as the owner sees them: all of them, or only those added after the
    first `since` = (bookings, messages) that were already there."""
    skip_bookings, skip_messages = since or (0, 0)
    bookings = (await runtime.store.list_bookings())[skip_bookings:]
    messages = (await runtime.store.list_messages())[skip_messages:]
    out("=== All saved records ===" if since is None else "=== Saved during this call ===")
    records: list[StoredBooking | StoredMessage] = [*bookings, *messages]
    if not records:
        out("(nothing)")
    for record in records:
        state = "отправлено" if record.notified_at else "не отправлено"
        out(f"--- Telegram: {state} ---")
        out(format_record(record))


async def _wait_for_delivery(runtime: Runtime, timeout: float, out: Out) -> None:
    deadline = time.monotonic() + timeout
    announced = False
    while await runtime.store.list_unnotified():
        if time.monotonic() > deadline:
            out(
                "(not all notifications were delivered; they stay in the database and are "
                "resent on the next --notify run)"
            )
            return
        if not announced:
            out("(waiting for Telegram delivery...)")
            announced = True
        await asyncio.sleep(0.2)
    if announced:
        out("(delivered)")


# --- Entry point ----------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        settings = get_settings()
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
        print(f"configuration error: {problems}", file=sys.stderr)
        return 1
    try:
        options = parse_options(args, settings)
    except UsageError as exc:
        parser.error(str(exc))  # prints usage + message, exits with status 2

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level, stream=sys.stderr, format="  [%(levelname)s] %(name)s: %(message)s"
    )

    async def _main() -> int:
        reader: ReadLine = (
            ScriptReader(options.script) if options.script is not None else StdinReader()
        )
        return await run(options, settings, reader)

    try:
        return asyncio.run(_main())
    except KeyboardInterrupt:
        print("\n(the caller hung up)")
        return 130
    except (StorageError, BusinessConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
