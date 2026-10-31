"""The terminal lock, and work postponed until no psnek lock is held."""

import logging
import os
import threading
from collections.abc import Callable
from functools import partial
from typing import Protocol

from psnek._text import sanitize

logger = logging.getLogger("psnek")

_pending: list[Callable[[], object]] = []


class _Depth(threading.local):
    """How many `CriticalLock` sections the current thread is inside."""

    value = 0


_depth = _Depth()


class _Lock(Protocol):
    def acquire(self, blocking: bool = ..., timeout: float = ...) -> bool: ...

    def release(self) -> None: ...


class CriticalLock:
    """A lock recording, per thread, that a psnek critical section is active."""

    __slots__ = ("lock",)

    def __init__(self, lock: _Lock) -> None:
        """Wrap the underlying ``lock``."""
        self.lock = lock

    def acquire(self, timeout: float = -1) -> bool:
        """Enter a critical section, then try to acquire the lock within ``timeout``."""
        depth = _depth.value
        _depth.value = depth + 1
        acquired = False
        try:
            acquired = self.lock.acquire(True, timeout)
        finally:
            if not acquired:
                _depth.value = depth
        return acquired

    def release(self) -> None:
        """Leave the critical section and release the lock."""
        try:
            _depth.value -= 1
        finally:
            self.lock.release()

    def __enter__(self) -> None:
        """Acquire the lock."""
        self.acquire()

    def __exit__(self, *exc_info: object) -> None:
        """Release the lock."""
        self.release()


_guard = CriticalLock(threading.RLock())
LOCK = CriticalLock(threading.RLock())


def busy() -> bool:
    """Return whether this thread holds a `CriticalLock`."""
    return _depth.value > 0


def defer_call(action: Callable[[], object]) -> None:
    """Queue the no-argument ``action`` to run at the next `run_deferred`."""
    with _guard:
        _pending.append(action)


def defer(level: int, message: str) -> None:
    """Queue a log record of ``message`` at ``level``."""
    defer_call(partial(logger.log, level, message))


def run_deferred() -> None:
    """Run the queued actions, oldest first, unless this thread is busy."""
    global _pending
    if not _pending or busy():
        return
    fresh: list[Callable[[], object]] = []
    with _guard:
        actions, _pending = _pending, fresh
    for position, action in enumerate(actions, 1):
        try:
            try:
                action()
            except Exception as error:
                _report(error)
        except BaseException:
            rest = actions[position:]
            with _guard:
                _pending[:0] = rest
            raise


def describe(error: BaseException) -> str:
    """Format ``error`` as sanitized ``Type: message``, or the type if that fails."""
    name = type(error).__name__
    try:
        message = str(error)
    except Exception:  # a broken __str__: the type name alone identifies it
        return name
    return sanitize(f"{name}: {message}") if message else name


def _report(error: Exception) -> None:
    message = f"psnek could not emit a deferred record: {describe(error)}"
    record = logger.makeRecord(logger.name, logging.ERROR, "", 0, message, (), None)
    try:
        (logging.lastResort or logging.StreamHandler()).handle(record)
    except Exception:
        return  # the last-resort channel itself failed: nothing is left to log to


def pending() -> bool:
    """Return whether `run_deferred` has queued work."""
    return bool(_pending)


def _after_fork_in_child() -> None:
    _guard.lock, LOCK.lock = threading.RLock(), threading.RLock()
    _pending.clear()
    _depth.value = 0


if hasattr(os, "register_at_fork"):
    os.register_at_fork(
        before=LOCK.acquire,
        after_in_parent=LOCK.release,
        after_in_child=_after_fork_in_child,
    )
