"""Offline tests for SqliteSink: round trips, durability, permissions, migrations, outbox."""

import asyncio
import os
import sqlite3
import stat
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from agent.business import load_business_config
from agent.llm import ToolCall
from agent.records import Booking, CallbackMessage, InMemorySink
from agent.storage import (
    SCHEMA_VERSION,
    SqliteSink,
    StorageError,
    StoredBooking,
    StoredMessage,
)
from agent.tools import ToolRegistry

MOSCOW = ZoneInfo("Europe/Moscow")
NOW = datetime(2026, 9, 24, 17, 5, 30, 123456, tzinfo=MOSCOW)
REPO_CONFIG = Path(__file__).parent.parent / "config" / "business.yaml"


def make_booking(**overrides) -> Booking:
    fields = {
        "name": "Игорь",
        "phone": "+79161234567",
        "car": "Тойота Камри",
        "service_id": "polishing",
        "service_name": "Полировка кузова",
        "preferred_date": date(2026, 9, 26),
        "preferred_time": time(14, 30),
        "preferred_period": None,
        "notes": None,
        "caller_phone": "+79991234567",
        "created_at": NOW,
    }
    fields.update(overrides)
    return Booking(**fields)


def make_message(**overrides) -> CallbackMessage:
    fields = {
        "message": "Хочу узнать про скидки",
        "name": "Анна",
        "phone": "+79161234567",
        "caller_phone": "+79991234567",
        "created_at": NOW,
    }
    fields.update(overrides)
    return CallbackMessage(**fields)


@pytest.fixture
def db_path(tmp_path) -> Path:
    return tmp_path / "data" / "agent.db"


@pytest.fixture
async def sink(db_path):
    sink = await SqliteSink.open(db_path)
    yield sink
    await sink.close()


# --- Round trips ---------------------------------------------------------------------------------


async def test_booking_round_trips_every_field(sink):
    booking = make_booking(notes="после обеда, «на пару часов»", preferred_period="день")

    booking_id = await sink.add_booking(booking)
    stored = await sink.get_booking(booking_id)

    assert stored == StoredBooking(booking_id, booking, None)
    assert stored.booking.created_at == NOW  # same instant, microseconds kept


async def test_booking_with_no_time_period_notes_or_caller_id_round_trips(sink):
    booking = make_booking(
        preferred_time=None, preferred_period=None, notes=None, caller_phone=None
    )
    booking_id = await sink.add_booking(booking)

    assert (await sink.get_booking(booking_id)).booking == booking


async def test_booking_with_period_and_exact_time_keeps_both(sink):
    booking = make_booking(preferred_time=time(9, 5), preferred_period="утро")
    booking_id = await sink.add_booking(booking)

    stored = (await sink.get_booking(booking_id)).booking
    assert stored.preferred_time == time(9, 5) and stored.preferred_period == "утро"


async def test_message_round_trips(sink):
    await sink.add_message(make_message())
    await sink.add_message(make_message(name=None, phone=None, caller_phone=None))

    stored = await sink.list_messages()

    assert [m.message for m in stored] == [
        make_message(),
        make_message(name=None, phone=None, caller_phone=None),
    ]
    assert all(isinstance(m, StoredMessage) for m in stored)


async def test_text_is_stored_verbatim_including_quotes_and_sql_metacharacters(sink):
    nasty = "Robert'); DROP TABLE bookings;-- «ёЁ» \n второй абзац"
    booking_id = await sink.add_booking(make_booking(name=nasty, notes=nasty))

    stored = (await sink.get_booking(booking_id)).booking
    assert stored.name == nasty and stored.notes == nasty
    assert len(await sink.list_bookings()) == 1


async def test_ids_increase_and_are_per_table(sink):
    assert await sink.add_booking(make_booking()) == 1
    assert await sink.add_booking(make_booking(car="BMW")) == 2
    assert await sink.add_message(make_message()) == 1  # a separate id space

    assert [b.id for b in await sink.list_bookings()] == [1, 2]


async def test_unknown_booking_id_returns_none(sink):
    assert await sink.get_booking(999) is None


async def test_created_at_keeps_its_utc_offset(sink):
    booking_id = await sink.add_booking(make_booking())
    created = (await sink.get_booking(booking_id)).booking.created_at
    assert created.utcoffset() == timedelta(hours=3)


# --- Durability -----------------------------------------------------------------------------------


async def test_data_survives_close_and_reopen(db_path):
    first = await SqliteSink.open(db_path)
    booking_id = await first.add_booking(make_booking())
    await first.add_message(make_message())
    await first.close()

    second = await SqliteSink.open(db_path)
    try:
        assert (await second.get_booking(booking_id)).booking == make_booking()
        assert len(await second.list_messages()) == 1
    finally:
        await second.close()


async def test_a_returned_id_means_the_row_is_already_visible_to_another_connection(sink, db_path):
    """Crash durability: no close(), no explicit commit by the caller."""
    await sink.add_booking(make_booking())

    other = sqlite3.connect(db_path)
    try:
        assert other.execute("SELECT COUNT(*) FROM bookings").fetchone()[0] == 1
    finally:
        other.close()


async def test_wal_and_full_synchronous_are_enabled(sink, db_path):
    other = sqlite3.connect(db_path)
    try:
        assert other.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        other.close()
    assert sink._conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL


async def test_a_failed_insert_leaves_nothing_behind(sink):
    good = make_booking()
    bad = make_booking(name=None)  # violates NOT NULL

    with pytest.raises(StorageError):
        await sink.add_booking(bad)

    assert await sink.list_bookings() == []
    assert await sink.add_booking(good) == 1  # and the sink is still usable


async def test_many_concurrent_inserts_are_all_stored_with_unique_ids(sink):
    ids = await asyncio.gather(*(sink.add_booking(make_booking(car=f"car {i}")) for i in range(60)))

    assert sorted(ids) == list(range(1, 61))
    assert {b.booking.car for b in await sink.list_bookings()} == {f"car {i}" for i in range(60)}


# --- File and directory ---------------------------------------------------------------------------


async def test_database_file_is_owner_only(sink, db_path):
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
    await sink.add_booking(make_booking())  # forces WAL sidecar files to exist
    for sidecar in db_path.parent.glob("agent.db-*"):
        assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600


async def test_loose_permissions_on_an_existing_file_are_tightened(tmp_path):
    path = tmp_path / "agent.db"
    path.touch()
    path.chmod(0o644)

    sink = await SqliteSink.open(path)
    await sink.close()

    assert stat.S_IMODE(path.stat().st_mode) == 0o600


async def test_missing_parent_directories_are_created_owner_only(db_path):
    sink = await SqliteSink.open(db_path)
    await sink.close()

    assert db_path.exists()
    assert stat.S_IMODE(db_path.parent.stat().st_mode) == 0o700


async def test_parent_that_is_a_file_raises_storage_error(tmp_path):
    (tmp_path / "data").write_text("not a directory")

    with pytest.raises(StorageError, match="cannot open"):
        await SqliteSink.open(tmp_path / "data" / "agent.db")


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
async def test_unwritable_directory_raises_storage_error(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(StorageError):
            await SqliteSink.open(locked / "agent.db")
    finally:
        locked.chmod(0o700)


async def test_a_file_that_is_not_a_database_raises_storage_error(tmp_path):
    path = tmp_path / "agent.db"
    path.write_bytes(b"this is definitely not sqlite" * 100)

    with pytest.raises(StorageError):
        await SqliteSink.open(path)


# --- Schema versioning ----------------------------------------------------------------------------


async def test_new_database_gets_the_current_schema_version(sink, db_path):
    other = sqlite3.connect(db_path)
    try:
        assert other.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        other.close()


async def test_reopening_keeps_data_and_does_not_migrate_twice(db_path):
    for _ in range(3):
        sink = await SqliteSink.open(db_path)
        await sink.add_booking(make_booking())
        await sink.close()

    sink = await SqliteSink.open(db_path)
    try:
        assert len(await sink.list_bookings()) == 3
    finally:
        await sink.close()


async def test_database_from_a_newer_version_is_refused_and_left_untouched(tmp_path):
    path = tmp_path / "agent.db"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 99")
    conn.execute("CREATE TABLE future (x)")
    conn.commit()
    conn.close()

    with pytest.raises(StorageError, match="newer than this code"):
        await SqliteSink.open(path)

    conn = sqlite3.connect(path)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables == {"future"}
    finally:
        conn.close()


# --- Closed sink ----------------------------------------------------------------------------------


async def test_operations_on_a_closed_sink_raise_storage_error(db_path):
    sink = await SqliteSink.open(db_path)
    await sink.close()

    with pytest.raises(StorageError, match="closed"):
        await sink.add_booking(make_booking())
    with pytest.raises(StorageError, match="closed"):
        await sink.list_unnotified()


async def test_close_is_idempotent_and_async_with_closes(db_path):
    async with await SqliteSink.open(db_path) as sink:
        await sink.add_booking(make_booking())
    await sink.close()  # second close is a no-op

    with pytest.raises(StorageError):
        await sink.list_bookings()


# --- Outbox ---------------------------------------------------------------------------------------


async def test_new_records_start_unnotified_and_are_listed_oldest_first(sink):
    later = NOW + timedelta(minutes=5)
    message_id = await sink.add_message(make_message(created_at=later))
    booking_id = await sink.add_booking(make_booking(created_at=NOW))

    pending = await sink.list_unnotified()

    assert [(type(r).__name__, r.id) for r in pending] == [
        ("StoredBooking", booking_id),
        ("StoredMessage", message_id),
    ]
    assert all(r.notified_at is None for r in pending)


async def test_marking_a_booking_notified_removes_it_from_the_outbox(sink):
    first = await sink.add_booking(make_booking())
    second = await sink.add_booking(make_booking(car="BMW", created_at=NOW + timedelta(seconds=1)))

    await sink.mark_booking_notified(first)

    assert [r.id for r in await sink.list_unnotified()] == [second]
    stored = await sink.get_booking(first)
    assert stored.notified_at is not None and stored.notified_at.tzinfo is not None


async def test_marking_a_message_notified_removes_it_from_the_outbox(sink):
    message_id = await sink.add_message(make_message())
    await sink.mark_message_notified(message_id)

    assert await sink.list_unnotified() == []
    assert (await sink.list_messages())[0].notified_at is not None


async def test_marking_uses_the_given_time_and_keeps_the_first_timestamp(sink):
    booking_id = await sink.add_booking(make_booking())
    first = datetime(2026, 9, 24, 12, 0, tzinfo=UTC)

    await sink.mark_booking_notified(booking_id, at=first)
    await sink.mark_booking_notified(booking_id, at=first + timedelta(hours=1))

    assert (await sink.get_booking(booking_id)).notified_at == first


async def test_marking_an_unknown_record_is_an_error(sink):
    with pytest.raises(StorageError, match="no such record"):
        await sink.mark_booking_notified(42)
    with pytest.raises(StorageError, match="no such record"):
        await sink.mark_message_notified(42)


async def test_the_outbox_survives_a_restart(db_path):
    first = await SqliteSink.open(db_path)
    sent = await first.add_booking(make_booking())
    lost = await first.add_booking(make_booking(car="BMW", created_at=NOW + timedelta(seconds=1)))
    await first.mark_booking_notified(sent)  # ...and the process dies before `lost` is notified
    await first.close()

    second = await SqliteSink.open(db_path)
    try:
        assert [r.id for r in await second.list_unnotified()] == [lost]
    finally:
        await second.close()


async def test_records_with_the_same_timestamp_are_ordered_by_id(sink):
    ids = [await sink.add_booking(make_booking(car=f"car {i}")) for i in range(3)]
    assert [r.id for r in await sink.list_unnotified()] == ids


# --- InMemorySink follows the same protocol -------------------------------------------------------


async def test_in_memory_sink_returns_ids_too():
    sink = InMemorySink()
    assert await sink.add_booking(make_booking()) == 1
    assert await sink.add_booking(make_booking()) == 2
    assert await sink.add_message(make_message()) == 1


# --- With the tools -------------------------------------------------------------------------------


def registry(sink) -> ToolRegistry:
    business = load_business_config(REPO_CONFIG)
    return ToolRegistry(business, sink, clock=lambda: NOW, caller_phone="+79991234567")


BOOKING_ARGS = (
    '{"name": "Игорь", "phone": "8 916 123 45 67", "car": "Тойота Камри",'
    ' "service_id": "polishing", "preferred_date": "2026-09-26", "preferred_period": "день",'
    ' "notes": "после обеда"}'
)


async def test_prepare_and_confirm_store_the_booking_in_sqlite(sink):
    tools = registry(sink)
    await tools.execute(ToolCall("c1", "prepare_booking", BOOKING_ARGS))
    assert await sink.list_bookings() == []  # nothing before confirm

    outcome = await tools.execute(ToolCall("c2", "confirm_booking", "{}"))

    assert not outcome.result.startswith("ОШИБКА")
    [stored] = await sink.list_bookings()
    assert stored.booking.phone == "+79161234567"
    assert stored.booking.preferred_period == "день" and stored.booking.preferred_time is None
    assert stored.booking.caller_phone == "+79991234567"
    assert stored.notified_at is None  # waiting for the notifier


async def test_take_message_stores_the_message_in_sqlite(sink):
    tools = registry(sink)
    await tools.execute(
        ToolCall("c1", "take_message", '{"message": "Перезвоните мне", "phone": "9161234567"}')
    )

    [stored] = await sink.list_messages()
    assert stored.message.message == "Перезвоните мне"
    assert stored.message.phone == "+79161234567"


async def test_a_storage_failure_is_reported_to_the_model_and_the_draft_survives(db_path):
    broken = await SqliteSink.open(db_path)
    tools = registry(broken)
    await tools.execute(ToolCall("c1", "prepare_booking", BOOKING_ARGS))
    await broken.close()  # the database goes away mid-call

    failed = await tools.execute(ToolCall("c2", "confirm_booking", "{}"))
    assert failed.result.startswith("ОШИБКА") and "не удалось" in failed.result

    recovered = await SqliteSink.open(db_path)  # e.g. the disk problem is fixed
    try:
        tools._sink = recovered
        again = await tools.execute(ToolCall("c3", "confirm_booking", "{}"))
        assert not again.result.startswith("ОШИБКА")
        assert len(await recovered.list_bookings()) == 1  # saved exactly once
    finally:
        await recovered.close()
