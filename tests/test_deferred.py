"""Tests for the deferred work queue and critical-section tracking."""

import io
import logging
import sys
import threading

import pytest

from psnek import _deferred
from tests.fakes import UnprintableError


def test_a_failing_action_does_not_stop_the_rest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    ran: list[str] = []

    def explode() -> None:
        raise LookupError("filter failed")

    _deferred.defer_call(explode)
    _deferred.defer_call(lambda: ran.append("second"))

    # Act
    _deferred.run_deferred()

    # Assert
    assert ran == ["second"]
    assert "filter failed" in capsys.readouterr().err


def test_nothing_runs_inside_a_critical_section() -> None:
    # Arrange
    lock = _deferred.CriticalLock(threading.Lock())
    ran: list[bool] = []
    _deferred.defer_call(lambda: ran.append(True))

    # Act
    with lock:
        busy = _deferred.busy()
        _deferred.run_deferred()
        during = list(ran)
    _deferred.run_deferred()

    # Assert
    assert busy
    assert during == []
    assert ran == [True]
    assert not _deferred.busy()


def test_deferred_records_reach_the_psnek_logger(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Act
    _deferred.defer(logging.WARNING, "queued message")
    logged_early = "queued message" in caplog.text
    _deferred.run_deferred()

    # Assert
    assert not logged_early
    assert "queued message" in caplog.text


class _InterruptedLock:
    """A lock whose acquisition is interrupted, as by Ctrl-C during the wait."""

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        raise KeyboardInterrupt

    def release(self) -> None:
        """Release nothing."""


def test_an_interrupted_entry_leaves_the_section() -> None:
    # Arrange
    lock = _deferred.CriticalLock(_InterruptedLock())

    # Act
    with pytest.raises(KeyboardInterrupt):
        lock.acquire()

    # Assert
    assert not _deferred.busy()
    assert _deferred._depth.value == 0


def test_a_closed_stderr_never_raises_into_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    closed = io.StringIO()
    closed.close()
    monkeypatch.setattr(logging, "lastResort", logging.StreamHandler(closed))
    monkeypatch.setattr(sys, "stderr", closed)
    monkeypatch.setattr(logging, "raiseExceptions", True)

    def explode() -> None:
        raise ValueError("I/O operation on closed file")

    _deferred.defer_call(explode)

    # Act
    _deferred.run_deferred()

    # Assert
    assert not _deferred.pending()


def test_an_interruption_after_earlier_actions_keeps_only_the_rest() -> None:
    # Arrange
    ran: list[str] = []

    def interrupt() -> None:
        raise KeyboardInterrupt

    for action in (
        lambda: ran.append("first"),
        interrupt,
        lambda: ran.append("kept"),
    ):
        _deferred.defer_call(action)

    # Act
    with pytest.raises(KeyboardInterrupt):
        _deferred.run_deferred()
    _deferred.run_deferred()

    # Assert
    assert ran == ["first", "kept"]


def test_an_interrupted_report_keeps_the_rest_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    ran: list[str] = []

    def interrupt(error: Exception) -> None:
        raise KeyboardInterrupt

    def explode() -> None:
        raise LookupError("handler failed")

    monkeypatch.setattr(_deferred, "_report", interrupt)
    _deferred.defer_call(explode)
    _deferred.defer_call(lambda: ran.append("kept"))

    # Act
    with pytest.raises(KeyboardInterrupt):
        _deferred.run_deferred()
    monkeypatch.undo()
    _deferred.run_deferred()

    # Assert
    assert ran == ["kept"]


def test_describe_names_the_type_even_when_the_message_fails() -> None:
    # Arrange
    cases = [
        (ValueError("bad"), "ValueError: bad"),
        (KeyboardInterrupt(), "KeyboardInterrupt"),
        (UnprintableError(), "UnprintableError"),
        (OSError("disk\r\x1b[2K full"), "OSError: disk\\u000d full"),
    ]

    # Act
    texts = [_deferred.describe(error) for error, _ in cases]

    # Assert
    assert texts == [expected for _, expected in cases]
