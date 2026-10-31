"""Process-wide output capture: ``sys.stdout``/``sys.stderr`` routed to the panel."""

import io
import logging
import sys
import threading
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, TextIO, cast

from psnek._deferred import LOCK
from psnek._text import clean_lines

if TYPE_CHECKING:
    from psnek._engine import Engine

CAPTURE_LIMIT, CAPTURE_PREVIEW, _ESCAPE_MARGIN = 1 << 20, 1024, 64
CAPTURING: "list[Engine]" = []
CAPTURES: "dict[str, Capture]" = {}


def _has_exited(thread: threading.Thread) -> bool:
    try:
        return not thread.is_alive()
    except RuntimeError:  # the stand-in for a foreign thread that ended (3.13+)
        return True


def _buffer(text: str) -> io.StringIO:
    buffer = io.StringIO()
    buffer.write(text)
    return buffer


def _tail(buffer: io.StringIO, size: int) -> str:
    """Read about the last ``size`` characters of ``buffer`` without copying it."""
    end = buffer.tell()
    buffer.seek(max(end - size - _ESCAPE_MARGIN, 0))
    text = buffer.read()
    cut = max(len(text) - size, 0)
    escape = text.rfind("\x1b", 0, cut)
    return text[escape:] if escape >= 0 else text[cut:]


class Capture:
    """Stand-in for ``sys.stdout``/``sys.stderr`` that routes lines to the panel."""

    def __init__(self, original: TextIO) -> None:
        """Wrap the replaced stream ``original`` (never a `Capture`)."""
        self.original = original
        self.pending: dict[threading.Thread, io.StringIO] = {}
        self.active = True

    def write(self, text: str) -> int:
        """Log complete lines of ``text``, buffering a partial line per thread."""
        if not isinstance(text, str):
            raise TypeError(f"write() argument must be str, not {type(text).__name__}")
        if not text:
            return 0
        with LOCK:
            if not self.active or not CAPTURING:
                return self.original.write(text)
            engine = CAPTURING[-1]
            lines = self._lines(threading.current_thread(), text)
        if lines:
            engine.log(clean_lines(lines))
        return len(text)

    def _lines(self, thread: threading.Thread, text: str) -> list[str]:
        """Add ``text`` to the thread's partial line; return completed lines."""
        buffer = self.pending.get(thread)
        if "\n" not in text:
            if buffer is None:
                buffer = self.pending[thread] = io.StringIO()
            buffer.write(text)
            if buffer.tell() <= CAPTURE_LIMIT:
                return []
            del self.pending[thread]
            return [buffer.getvalue()]
        joined = text if buffer is None else buffer.getvalue() + text
        *lines, rest = joined.split("\n")
        if rest and len(rest) <= CAPTURE_LIMIT:
            self.pending[thread] = _buffer(rest)
            return lines
        self.pending.pop(thread, None)
        return [*lines, rest] if rest else lines

    def writelines(self, lines: Iterable[str]) -> None:
        """Write each string in ``lines``, like ``io.TextIOBase.writelines``."""
        self.write("".join(lines))

    def preview(self) -> list[str]:
        """Return the last `CAPTURE_PREVIEW` characters of each partial line."""
        return [_tail(buffer, CAPTURE_PREVIEW) for buffer in self.pending.values()]

    def reap(self) -> list[str]:
        """Remove and return the unterminated text of threads that have exited."""
        finished = [thread for thread in self.pending if _has_exited(thread)]
        return [self.pending.pop(thread).getvalue() for thread in finished]

    def drain(self) -> list[str]:
        """Stop capturing and return the unterminated text of every thread."""
        with LOCK:
            self.active = False
            leftovers = [buffer.getvalue() for buffer in self.pending.values()]
            self.pending.clear()
            return leftovers

    def isatty(self) -> bool:
        """Return False while capturing, so libraries do not move the cursor."""
        return not self.active and self.original.isatty()

    def __getattr__(self, name: str) -> object:
        """Return the wrapped stream's attribute ``name``."""
        return getattr(self.original, name)


class _Discard(io.TextIOBase):
    """Write target used when the process has no ``sys.stderr``."""

    def write(self, text: str) -> int:
        """Discard ``text``, returning its length."""
        return len(text)


DISCARD = cast("TextIO", _Discard())


def unwrap(stream: TextIO) -> TextIO:
    """Return the stream behind any `Capture` proxies around ``stream``."""
    while isinstance(stream, Capture):
        stream = stream.original
    return stream


def stream_handlers() -> Iterator[logging.StreamHandler[TextIO]]:
    """Yield the live stream handlers whose stream can be reassigned."""
    references = getattr(logging, "_handlerList", None)
    if references is None:
        loggers = [logging.getLogger(), *logging.Logger.manager.loggerDict.values()]
        handlers = [one for item in loggers for one in getattr(item, "handlers", ())]
    else:
        handlers = [reference() for reference in references.copy()]
    for handler in handlers:
        if (
            isinstance(handler, logging.StreamHandler)
            and hasattr(handler, "stream")
            and not isinstance(getattr(type(handler), "stream", None), property)
        ):
            yield handler


def _restore_streams() -> None:
    for name, capture in CAPTURES.items():
        if getattr(sys, name) is capture:
            setattr(sys, name, capture.original)
    for handler in stream_handlers():
        if isinstance(handler.stream, Capture):
            handler.stream = unwrap(handler.stream)
    CAPTURES.clear()
