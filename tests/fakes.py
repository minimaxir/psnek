"""Test doubles shared by several test modules."""

import html
import io
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from psnek import Bar, Options, _deferred, _engine
from psnek._pace import WARMUP_GAPS
from tests.vt import Screen

QUIET = Options(disable=True)


@contextmanager
def engine_lock() -> Iterator[None]:
    """Hold the engine lock, then run what psnek deferred while it was held.

    psnek defers its log records while the lock is held; a test holding it
    drains them afterwards, as the code releasing the lock would.

    Yields:
        Nothing; the lock is held inside the ``with`` block.
    """
    try:
        with _engine.LOCK:
            yield
    finally:
        _deferred.run_deferred()


def engine_for(stream: io.StringIO) -> _engine.Engine:
    """Return the engine drawing to ``stream``."""
    return next(
        engine for engine in _engine.ENGINES.values() if engine.stream is stream
    )


def settle_idle_engines() -> None:
    """Do what the next render tick does for engines whose bars all ended."""
    with engine_lock():
        for engine in list(_engine.ENGINES.values()):
            if not engine.sources:
                engine.close_idle()


_TAG = re.compile(r"<br>|<[^>]*>")


def shown_text(update: dict[str, str]) -> str:
    """Return the text a notebook update shows, read from its HTML."""
    rows = _TAG.sub(lambda tag: "\n" * (tag.group() == "<br>"), update["text/html"])
    return html.unescape(rows)


def take_steps(
    bar: Bar, start: float, gaps: tuple[float, ...] = (0.5,) * WARMUP_GAPS
) -> float:
    """Count one unit per gap after ``start``, past the warm-up by default.

    The caller holds the engine lock.

    Args:
        bar: The bar to advance.
        start: Monotonic time the first gap starts from.
        gaps: Seconds between successive units.

    Returns:
        The monotonic time of the last unit.
    """
    for gap in gaps:
        start += gap
        bar.update()
        bar._tally(start)
    return start


class FakeHandle:
    """Records the updates sent to one notebook display."""

    def __init__(self) -> None:
        """Start with no updates."""
        self.updates: list[dict[str, str]] = []

    def update(self, obj: dict[str, str], *, raw: bool) -> None:
        """Record ``obj``."""
        self.updates.append(obj)


class FakeEvents:
    """The part of IPython's event manager psnek uses, run by hand."""

    def __init__(self) -> None:
        """Start with no callbacks."""
        self.callbacks: dict[str, list[Callable[[object], None]]] = {
            "pre_run_cell": [],
            "post_run_cell": [],
        }

    def register(self, event: str, function: Callable[[object], None]) -> None:
        """Register ``function`` to run on ``event``.

        Args:
            event: The event's name.
            function: Its callback.
        """
        self.callbacks[event].append(function)

    def start_cell(self) -> None:
        """Run the callbacks of a cell about to run, as IPython does."""
        for callback in self.callbacks["pre_run_cell"]:
            callback(None)

    def end_cell(self) -> None:
        """Run the callbacks of a cell that finished, as IPython does."""
        for callback in self.callbacks["post_run_cell"]:
            callback(None)


def screen_text(stream: io.StringIO) -> list[str]:
    """Replay everything written to ``stream`` on a screen of the environment's size.

    Args:
        stream: The stream psnek wrote to.

    Returns:
        The screen's rows, scrollback included, as `Screen.text` gives them.
    """
    screen = Screen(int(os.environ["COLUMNS"]), int(os.environ["LINES"]))
    screen.feed(stream.getvalue())
    return screen.text()


class FakeTerminal(io.StringIO):
    """An in-memory stream that claims to be a TTY."""

    def isatty(self) -> bool:
        """Pretend to be interactive.

        Returns:
            Always True.
        """
        return True


class _FailingWrites:
    """A mixin for in-memory streams whose every write raises ``error``."""

    def __init__(self, error: BaseException) -> None:
        """Fail writes with ``error``.

        Args:
            error: The exception every write raises.
        """
        super().__init__()
        self.error = error

    def write(self, data: object) -> int:
        """Fail the write.

        Args:
            data: Ignored text or bytes.

        Raises:
            BaseException: `error`, always.
        """
        raise self.error


class CountingStdout(io.StringIO):
    """An in-memory stdout that counts its flushes instead of flushing."""

    def __init__(self) -> None:
        """Start with no flushes counted."""
        super().__init__()
        self.flushes = 0

    def flush(self) -> None:
        """Count the flush."""
        self.flushes += 1


class FailingStream(_FailingWrites, io.StringIO):
    """A text stream whose every write raises ``error``."""


class FailingBytes(_FailingWrites, io.BytesIO):
    """A byte stream whose every write raises ``error``."""


class FlakyTerminal(FakeTerminal):
    """A TTY whose first write after `fail_next` is set raises `error`."""

    fail_next = False
    error: OSError = OSError("transient failure")

    def write(self, text: str) -> int:
        """Fail once when armed, otherwise record the text.

        Args:
            text: Text to record.

        Returns:
            The number of characters written.

        Raises:
            OSError: `error`, when armed.
        """
        if self.fail_next:
            self.fail_next = False
            raise self.error
        return super().write(text)


class UnprintableError(Exception):
    """An exception whose message cannot be built."""

    def __str__(self) -> str:
        """Fail to describe the error.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError("no message")
