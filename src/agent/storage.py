"""SQLite storage for bookings and callback messages: a `RecordSink` that survives crashes.

Everything the agent promises the business goes through here first. Step 1.8's Telegram
notifier runs *after* a successful save and then marks the row as notified, so nothing is lost
if Telegram (or the process) fails in between: rows with `notified_at IS NULL` are the outbox.

stdlib `sqlite3` on one shared connection, every call run in a worker thread
(`asyncio.to_thread`) under a lock, so the event loop never blocks on disk. WAL journal plus
`synchronous=FULL`: a row that `add_*` returned for is on disk. The file holds names and phone
numbers, so it is created with mode 0600.
"""

import asyncio
import logging
import os
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

from agent.records import Booking, CallbackMessage

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """The database is missing, unusable, closed, or refused an operation."""


# Schema migrations, applied in order; PRAGMA user_version records how many have been applied.
# To change the schema, append a script here (e.g. step 3 adds a call_id column) - never edit
# an existing entry.
_MIGRATIONS = (
    """
    CREATE TABLE bookings (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        phone TEXT NOT NULL,
        car TEXT NOT NULL,
        service_id TEXT NOT NULL,
        service_name TEXT NOT NULL,
        preferred_date TEXT NOT NULL,
        preferred_time TEXT,
        preferred_period TEXT,
        notes TEXT,
        caller_phone TEXT,
        created_at TEXT NOT NULL,
        notified_at TEXT
    );
    CREATE TABLE messages (
        id INTEGER PRIMARY KEY,
        message TEXT NOT NULL,
        name TEXT,
        phone TEXT,
        caller_phone TEXT,
        created_at TEXT NOT NULL,
        notified_at TEXT
    );
    CREATE INDEX bookings_unnotified ON bookings (id) WHERE notified_at IS NULL;
    CREATE INDEX messages_unnotified ON messages (id) WHERE notified_at IS NULL;
    """,
)
SCHEMA_VERSION = len(_MIGRATIONS)

DB_FILE_MODE = 0o600
DB_DIR_MODE = 0o700


@dataclass(frozen=True)
class StoredBooking:
    id: int
    booking: Booking
    notified_at: datetime | None


@dataclass(frozen=True)
class StoredMessage:
    id: int
    message: CallbackMessage
    notified_at: datetime | None


class SqliteSink:
    """A RecordSink backed by one SQLite file. Create it with `await SqliteSink.open(path)`."""

    def __init__(self, path: Path, conn: sqlite3.Connection) -> None:
        self._path = path
        self._conn: sqlite3.Connection | None = conn
        self._lock = threading.Lock()

    # --- lifecycle ------------------------------------------------------------------------

    @classmethod
    async def open(cls, path: Path) -> "SqliteSink":
        return await asyncio.to_thread(cls._open_sync, Path(path))

    @classmethod
    def _open_sync(cls, path: Path) -> "SqliteSink":
        try:
            if not path.parent.exists():
                path.parent.mkdir(parents=True, mode=DB_DIR_MODE)
            # Create the file ourselves so it never exists with looser permissions; SQLite
            # gives the -wal/-shm sidecar files the same mode as the main file.
            os.close(os.open(path, os.O_RDWR | os.O_CREAT, DB_FILE_MODE))
            os.chmod(path, DB_FILE_MODE)
            conn = sqlite3.connect(path, check_same_thread=False, timeout=5.0)
        except (OSError, sqlite3.Error) as exc:
            raise StorageError(f"cannot open database {path}: {exc}") from exc

        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            cls._migrate(conn, path)
        except sqlite3.Error as exc:
            conn.close()
            raise StorageError(f"cannot use database {path}: {exc}") from exc
        except StorageError:
            conn.close()
            raise
        return cls(path, conn)

    @staticmethod
    def _migrate(conn: sqlite3.Connection, path: Path) -> None:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise StorageError(
                f"database {path} has schema version {version}, newer than this code "
                f"understands ({SCHEMA_VERSION}); refusing to touch it"
            )
        for number in range(version, SCHEMA_VERSION):
            # executescript commits any open transaction first, so each migration is its own
            # atomic BEGIN..COMMIT together with the version bump.
            conn.executescript(
                f"BEGIN;\n{_MIGRATIONS[number]}\nPRAGMA user_version = {number + 1};\nCOMMIT;"
            )
            logger.info("database %s migrated to schema version %d", path, number + 1)

    async def close(self) -> None:
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    async def __aenter__(self) -> "SqliteSink":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # --- RecordSink -----------------------------------------------------------------------

    async def add_booking(self, booking: Booking) -> int:
        return await self._run(self._insert_booking, booking)

    async def add_message(self, message: CallbackMessage) -> int:
        return await self._run(self._insert_message, message)

    # --- Reading ----------------------------------------------------------------------------

    async def get_booking(self, booking_id: int) -> StoredBooking | None:
        return await self._run(self._select_booking, booking_id)

    async def list_bookings(self) -> list[StoredBooking]:
        return await self._run(self._select_bookings)

    async def list_messages(self) -> list[StoredMessage]:
        return await self._run(self._select_messages)

    # --- Outbox (used by the notifier, step 1.8) ------------------------------------------------

    async def list_unnotified(self) -> list[StoredBooking | StoredMessage]:
        """Saved records nobody has been told about yet, oldest first."""
        return await self._run(self._select_unnotified)

    async def mark_booking_notified(self, booking_id: int, at: datetime | None = None) -> None:
        await self._run(self._mark, "bookings", booking_id, at)

    async def mark_message_notified(self, message_id: int, at: datetime | None = None) -> None:
        await self._run(self._mark, "messages", message_id, at)

    # --- Internals -----------------------------------------------------------------------------

    async def _run(self, fn: Callable[..., Any], *args: Any) -> Any:
        return await asyncio.to_thread(self._locked, fn, *args)

    def _locked(self, fn: Callable[..., Any], *args: Any) -> Any:
        with self._lock:
            if self._conn is None:
                raise StorageError("storage is closed")
            try:
                with self._conn:  # commits on success, rolls back on error
                    return fn(self._conn, *args)
            except sqlite3.Error as exc:
                raise StorageError(f"database error: {exc}") from exc

    @staticmethod
    def _insert_booking(conn: sqlite3.Connection, booking: Booking) -> int:
        cursor = conn.execute(
            "INSERT INTO bookings (name, phone, car, service_id, service_name, preferred_date,"
            " preferred_time, preferred_period, notes, caller_phone, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                booking.name,
                booking.phone,
                booking.car,
                booking.service_id,
                booking.service_name,
                booking.preferred_date.isoformat(),
                booking.preferred_time.strftime("%H:%M") if booking.preferred_time else None,
                booking.preferred_period,
                booking.notes,
                booking.caller_phone,
                booking.created_at.isoformat(),
            ),
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    @staticmethod
    def _insert_message(conn: sqlite3.Connection, message: CallbackMessage) -> int:
        cursor = conn.execute(
            "INSERT INTO messages (message, name, phone, caller_phone, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                message.message,
                message.name,
                message.phone,
                message.caller_phone,
                message.created_at.isoformat(),
            ),
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    @classmethod
    def _select_booking(cls, conn: sqlite3.Connection, booking_id: int) -> StoredBooking | None:
        row = conn.execute("SELECT * FROM bookings WHERE id = ?", (booking_id,)).fetchone()
        return _booking_from_row(row) if row else None

    @classmethod
    def _select_bookings(cls, conn: sqlite3.Connection) -> list[StoredBooking]:
        rows = conn.execute("SELECT * FROM bookings ORDER BY id").fetchall()
        return [_booking_from_row(row) for row in rows]

    @classmethod
    def _select_messages(cls, conn: sqlite3.Connection) -> list[StoredMessage]:
        rows = conn.execute("SELECT * FROM messages ORDER BY id").fetchall()
        return [_message_from_row(row) for row in rows]

    @classmethod
    def _select_unnotified(cls, conn: sqlite3.Connection) -> list[StoredBooking | StoredMessage]:
        bookings = conn.execute("SELECT * FROM bookings WHERE notified_at IS NULL").fetchall()
        messages = conn.execute("SELECT * FROM messages WHERE notified_at IS NULL").fetchall()
        stored: list[StoredBooking | StoredMessage] = [_booking_from_row(r) for r in bookings]
        stored += [_message_from_row(r) for r in messages]
        # Oldest first; the id breaks ties between records created at the same instant.
        return sorted(stored, key=_sort_key)

    @staticmethod
    def _mark(conn: sqlite3.Connection, table: str, record_id: int, at: datetime | None) -> None:
        # `table` is one of two literals from the methods above, never caller input.
        when = (at or datetime.now(UTC)).isoformat()
        if not conn.execute(f"SELECT 1 FROM {table} WHERE id = ?", (record_id,)).fetchone():
            raise StorageError(f"no such record: {table} id {record_id}")
        # Marking twice is harmless and keeps the first timestamp.
        conn.execute(
            f"UPDATE {table} SET notified_at = ? WHERE id = ? AND notified_at IS NULL",
            (when, record_id),
        )


def _sort_key(record: StoredBooking | StoredMessage) -> tuple[datetime, int]:
    if isinstance(record, StoredBooking):
        return record.booking.created_at, record.id
    return record.message.created_at, record.id


def _parse_notified(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _booking_from_row(row: sqlite3.Row) -> StoredBooking:
    return StoredBooking(
        id=row["id"],
        booking=Booking(
            name=row["name"],
            phone=row["phone"],
            car=row["car"],
            service_id=row["service_id"],
            service_name=row["service_name"],
            preferred_date=date.fromisoformat(row["preferred_date"]),
            preferred_time=time.fromisoformat(row["preferred_time"])
            if row["preferred_time"]
            else None,
            preferred_period=row["preferred_period"],
            notes=row["notes"],
            caller_phone=row["caller_phone"],
            created_at=datetime.fromisoformat(row["created_at"]),
        ),
        notified_at=_parse_notified(row["notified_at"]),
    )


def _message_from_row(row: sqlite3.Row) -> StoredMessage:
    return StoredMessage(
        id=row["id"],
        message=CallbackMessage(
            message=row["message"],
            name=row["name"],
            phone=row["phone"],
            caller_phone=row["caller_phone"],
            created_at=datetime.fromisoformat(row["created_at"]),
        ),
        notified_at=_parse_notified(row["notified_at"]),
    )
