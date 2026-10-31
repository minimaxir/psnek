"""The manual progress bar, its options, `pmap`, and text output helpers."""

import logging
import math
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence, Sized
from concurrent.futures import (
    FIRST_EXCEPTION,
    Executor,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from functools import partial, wraps
from itertools import chain, repeat, starmap
from operator import length_hint
from typing import Literal, Self, TextIO, TypeVar, get_args

from psnek import _engine
from psnek._backends import TICK
from psnek._deferred import LOCK, busy, defer_call, describe, run_deferred
from psnek._env import once
from psnek._frame import Frame, Outcome, State
from psnek._pace import Pace
from psnek._render import log_lines, log_stamp
from psnek._style import parse_hex
from psnek._table import Justify, as_record, panel_lines
from psnek._text import SCALES, Scale, format_value

T = TypeVar("T")
R = TypeVar("R")
FieldValue = str | int | float | None
PanelPosition = Literal["below", "above"]

CHUNKS_PER_WORKER = 4
_SLICEABLE = (list, tuple, range, str, bytes)
_ANIMATION_SPEED: dict[str, float] = {"running": 1.0, "stalled": 0.35, "stuck": 0.15}
_NOT_NUMBERS = (str, bytes, bytearray, bool)
_FLAGS = ("panel_clear", "log_time", "capture", "notify", "disable")
_THREAD = threading.local()
_CAUSES = {
    ("builtins", "KeyboardInterrupt"): "Ctrl-C",
    ("builtins", "SystemExit"): "exit",
    ("builtins", "GeneratorExit"): "closed",
    ("asyncio.exceptions", "CancelledError"): "cancelled",
}


def _ignore(n: float) -> None:
    """Stand in for ``update`` once a bar has finished."""


def _gil_enabled() -> bool:
    check = getattr(sys, "_is_gil_enabled", None)
    return check is None or bool(check())


# The GIL can be re-enabled at runtime but never disabled, so this stays valid.
_ONE_QUEUE = _gil_enabled()


def _thread_queue(n: float) -> None:
    """Mark a bar whose updates go to a queue per thread (never called)."""


def _owner() -> object:
    asyncio = sys.modules.get("asyncio")
    try:
        task = None if asyncio is None else asyncio.current_task()
    except RuntimeError:  # no event loop runs in this thread
        task = None
    return task or threading.current_thread()


def _lines(values: tuple[object, ...], sep: str) -> list[str]:
    return sep.join(map(str, values)).split("\n")


@dataclass(frozen=True, slots=True, kw_only=True)
class Options:
    """Presentation settings shared by every bar type.

    Attributes:
        unit: Unit label, e.g. ``"it"``, ``"B"``, ``"file"``.
        scale: ``"si"`` (1000) or ``"iec"`` (1024) number prefixes, or ``"none"``.
        leave: Keep the final line after closing; None keeps only the top-level bar.
        stream: Output stream (default ``sys.stderr``).
        panel_lines: Maximum panel text lines (panel settings follow a first bar).
        panel_position: Draw the panel ``"below"`` or ``"above"`` the bars.
        panel_justify: Align `Bar.panel` columns: ``"left"``, ``"right"``, ``"center"``.
        panel_clear: Remove the `Bar.panel` content when the bars finish.
        log_time: Start each `Bar.log` line with the local time (``HH:MM:SS``).
        capture: In live mode, route ``print``/logging output into the panel.
        notify: Send desktop notifications on terminals that support them.
        disable: Draw no bar (progress and text still work).
        height: Rows the bar spans, wrapping its description, with its status below.
        gradient: Color running and finished bars with the palette gradient, with a
            list or tuple of ``"#RRGGBB"`` colors spread from left to right, or
            (False) in the default foreground color (state colors still apply).

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If ``scale``, a panel setting, ``height``, or a color is invalid.
    """

    unit: str = "it"
    scale: Scale = "none"
    leave: bool | None = None
    stream: TextIO | None = None
    panel_lines: int = 8
    panel_position: PanelPosition = "below"
    panel_justify: Justify = "left"
    panel_clear: bool = False
    log_time: bool = False
    capture: bool = True
    notify: bool = True
    disable: bool = False
    height: int = 1
    gradient: bool | tuple[str, ...] | list[str] = True
    _stops: bool | tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the settings."""
        if not isinstance(self.unit, str):
            raise TypeError("unit must be a string")
        if type(self.panel_lines) is not int or type(self.height) is not int:
            raise TypeError("panel_lines and height must be ints")
        if not all(isinstance(getattr(self, name), bool) for name in _FLAGS):
            raise TypeError(f"{', '.join(_FLAGS)} must be bools")
        if self.leave is not None and not isinstance(self.leave, bool):
            raise TypeError("leave must be a bool or None")
        if self.stream is not None and not callable(getattr(self.stream, "write", 0)):
            raise TypeError("stream must be a text stream with a write method")
        gradient = self.gradient
        if isinstance(gradient, list | tuple) and all(
            isinstance(c, str) for c in gradient
        ):
            gradient = tuple(gradient)
        elif not isinstance(gradient, bool):
            raise TypeError("gradient must be a bool or a list or tuple of hex strings")
        for name, value, allowed in (
            ("scale", self.scale, SCALES),
            ("panel_position", self.panel_position, get_args(PanelPosition)),
            ("panel_justify", self.panel_justify, get_args(Justify)),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}")
        if self.panel_lines < 0 or self.height < 1:
            raise ValueError("panel_lines must be non-negative and height at least 1")
        stops = gradient if isinstance(gradient, bool) else parse_hex(gradient)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "_stops", stops)


def ending(exc_type: type[BaseException] | None) -> tuple[Outcome | None, str]:
    """Map ``exc_type`` leaving a bar to its outcome and the cause shown with it.

    Example:
        >>> ending(KeyboardInterrupt), ending(ValueError)
        (('interrupted', 'Ctrl-C'), ('failed', 'ValueError'))
    """
    if exc_type is None:
        return None, ""
    outcome: Outcome = "failed" if issubclass(exc_type, Exception) else "interrupted"
    cause = _CAUSES.get((exc_type.__module__, exc_type.__qualname__))
    return outcome, cause or exc_type.__name__


_DEFAULT_OPTIONS = Options()


def _shown_engine() -> _engine.Engine:
    """Return the engine showing bars (else the default stream's), stdout flushed."""
    with LOCK:
        engine = _engine.active_engine() or _engine.find(_DEFAULT_OPTIONS)
    _engine.flush_stdout(engine.stream)
    return engine


class Bar:
    """A progress bar for manually reported work, shown from creation until `close`.

    Attributes:
        leave: Whether the final line is kept after closing (None: automatic).
        notify: Whether desktop notifications may report this bar.

    Example:
        >>> with Bar(total=3, desc="upload", options=Options(disable=True)) as bar:
        ...     bar.update(3)
        >>> bar.n
        3
    """

    # fmt: off
    __slots__ = (
        "__weakref__", "_cause", "_desc", "_draining", "_engine", "_exhausted",
        "_failed", "_fetched", "_initial", "_iterated", "_latest", "_managed",
        "_manual", "_options", "_origin", "_outcome", "_pace", "_phase", "_push",
        "_queues", "_settled", "_sized", "_status", "_total", "leave", "notify",
    )
    # fmt: on
    _wraps_iterable = False

    def __init__(
        self,
        total: float | None = None,
        desc: object = "",
        *,
        options: Options | None = None,
    ) -> None:
        """Create a bar; manual bars are displayed immediately.

        Args:
            total: Expected number of units, or None if unknown.
            desc: Short description shown before the bar (converted with ``str``).
            options: Presentation settings.

        Raises:
            TypeError: If ``total`` is not a number or ``options`` not `Options`.
            ValueError: If ``total`` is negative or not finite.
        """
        if options is not None and not isinstance(options, Options):
            raise TypeError(f"options must be an Options, not {type(options).__name__}")
        self._options = _DEFAULT_OPTIONS if options is None else options
        self._total = _validate_total(total)
        self._desc = str(desc)
        self.leave, self.notify = self._options.leave, self._options.notify
        self._status: tuple[tuple[str, str], ...] = ()
        self._queues: list[deque[float]] = [deque()] if _ONE_QUEUE else []
        self._push = self._queues[0].append if _ONE_QUEUE else _thread_queue
        self._manual: float = 0
        self._initial: float = 0
        self._fetched: Callable[[], int] = int
        self._exhausted = self._settled = self._managed = self._draining = False
        self._failed = self._iterated = False
        self._cause = ""
        self._sized: Sized | None = None
        self._outcome: Outcome | None = None
        self._engine: _engine.Engine | None = None
        run_deferred()
        self._start(time.monotonic())
        if not self._wraps_iterable:
            self._target()

    def _start(self, now: float) -> None:
        count = self._count()
        self._origin, self._pace = now, Pace(count, now)
        self._latest, self._phase = (count, 0.0), 0.0

    def _close_dropped(self) -> None:
        """Close an unmanaged bar whose loop was abandoned, from a GC finalizer."""
        if self._managed or self._outcome is not None:
            return
        if not busy() and self._engine is not None:
            _engine.flush_stdout(self._engine.stream)
        if busy() or not LOCK.acquire(timeout=TICK):
            defer_call(self.close)
            return
        try:
            final = self._conclude(None)
        finally:
            LOCK.release()
        if final is not None:
            run_deferred()

    def _count_pulls(self) -> "repeat[bool]":
        """Return an endless selector; the items taken from it are fetched items."""
        selector = repeat(True, sys.maxsize)
        left = partial(length_hint, selector)
        self._fetched = lambda: sys.maxsize - left()
        return selector

    def _end_loop(self, outcome: Outcome | None, cause: str = "") -> None:
        """Finish an unmanaged bar when its loop ends; ``with`` does it otherwise."""
        if not self._managed:
            self._finish(outcome, cause)

    def _target(self, claim: bool = False) -> None:
        """Join the engine for this bar's stream, starting the bar's clock."""
        if self._engine is not None and not claim:
            return
        options = self._options
        _engine.flush_stdout(sys.stderr if options.stream is None else options.stream)
        with LOCK:
            if claim:
                iterated, self._iterated = self._iterated, True
                if iterated:
                    raise RuntimeError("a psnek bar can only be iterated once")
            if self._engine is not None:
                return
            engine = _engine.find(options)
            if self._outcome is None:
                now = time.monotonic()
                if self._wraps_iterable:
                    self._start(now)
                engine.add(self, options, now, _owner())
            self._engine = engine

    def _count(self) -> float:
        """Return completed units; the caller holds `LOCK`."""
        if self._outcome is not None:
            for queue in self._queues:
                queue.clear()
        elif not self._draining and any(self._queues):
            self._draining = True
            try:
                pulls = chain.from_iterable(
                    starmap(queue.popleft, repeat((), len(queue)))
                    for queue in self._queues
                )
                self._manual = _accumulate(self._manual, pulls, self._initial)
            finally:
                self._draining = False
        fetched = self._fetched()
        iterated = fetched if self._settled else max(fetched - 1, 0)
        return self._initial + iterated + self._manual

    def _sample(
        self, now: float, branches: tuple[bool, ...] = (), children: bool = False
    ) -> Frame:
        count = self._count()
        if self._sized is not None:
            self._total = len(self._sized)
        total, pace, options = self._total, self._pace, self._options
        elapsed = now - self._origin
        step = elapsed - self._latest[1]
        pace.advance(count, now)
        estimate, approximate = None, False
        if self._outcome is not None:
            state: State = self._outcome
        elif total is not None and count >= total:
            state = "running"
        else:
            state = pace.health(now)
            if total is not None and pace.rate is not None:
                estimate, approximate = pace.eta(total - count, now)
            approximate = approximate or state != "running"
        self._phase += max(step, 0.0) * _ANIMATION_SPEED.get(state, 0.0)
        self._latest = (count, elapsed)
        return Frame(
            n=count,
            total=total,
            elapsed=elapsed,
            rate=pace.rate,
            desc=self._desc,
            unit=options.unit,
            scale=options.scale,
            status=self._status,
            state=state,
            idle=now - pace.changed_at,
            phase=self._phase,
            branches=branches,
            estimate=estimate,
            approximate=approximate,
            height=options.height,
            children=children,
            cause=self._cause,
            gradient=options._stops,
        )

    def _tally(self, now: float) -> None:
        count = self._count()
        self._pace.advance(count, now)
        self._latest = (count, now - self._origin)

    @property
    def desc(self) -> str:
        """Short description shown before the bar; assigned values become ``str``."""
        return self._desc

    @desc.setter
    def desc(self, value: object) -> None:
        self._desc = str(value)

    @property
    def n(self) -> float:
        """Units completed so far; fixed once the bar has finished."""
        with LOCK:
            return self._count() if self._outcome is None else self._latest[0]

    @property
    def total(self) -> float | None:
        """Expected units, or None; follows a wrapped list's length until set."""
        sized = self._sized
        return self._total if sized is None else len(sized)

    @total.setter
    def total(self, value: float | None) -> None:
        validated = _validate_total(value)
        with LOCK:
            self._total, self._sized = validated, None

    @property
    def rate(self) -> float | None:
        """Smoothed units per second as of the last frame."""
        return self._pace.rate

    @property
    def elapsed(self) -> float:
        """Seconds since the bar started; fixed once closed."""
        if self._outcome is not None:
            return self._latest[1]
        return time.monotonic() - self._origin

    @property
    def outcome(self) -> Outcome | None:
        """How the bar finished (``done``, ``failed``, ...), or None while open."""
        return self._outcome

    def update(self, n: float = 1) -> None:
        """Record ``n`` more completed units. Safe to call from any thread.

        Args:
            n: Units completed since the last call (may be negative).
        """
        self._push(n)

    def _thread_push(self) -> Callable[[float], None]:
        """Give the calling thread its own queue of increments; return its append."""
        queue: deque[float] = deque()
        self._queues.append(queue)
        pushes: dict[Bar, Callable[[float], None]]
        pushes = _THREAD.__dict__.setdefault("pushes", {})
        for closed in [bar for bar in pushes if bar._outcome is not None]:
            del pushes[closed]
        return pushes.setdefault(self, queue.append)

    def status(self, values: object = None, /, **fields: FieldValue) -> None:
        """Merge ``key=value`` fields shown after the metrics; None removes a key.

        Args:
            values: A mapping, dataclass instance, or named tuple of fields.
            **fields: More fields, taking precedence over ``values``.

        Raises:
            TypeError: If ``values`` is not a mapping, dataclass, or named tuple.

        Example:
            >>> with Bar(total=2, options=Options(disable=True)) as bar:
            ...     bar.status({"loss": 0.1376, "val loss": 0.21}, lr=3e-4)
            ...     bar.status(lr=None)
        """
        record = {} if values is None else as_record(values)
        if record is None:
            raise TypeError("status needs a mapping, dataclass, or named tuple")
        shown = {
            str(key): None if value is None else format_value(value)
            for key, value in chain(record.items(), fields.items())
        }
        with LOCK:
            merged = dict(self._status) | shown
            self._status = tuple(
                (key, text) for key, text in merged.items() if text is not None
            )

    def _show(self, action: Callable[[_engine.Engine], None]) -> None:
        with LOCK:
            engine = self._engine or _engine.find(self._options)
        _engine.flush_stdout(engine.stream)
        action(engine)
        run_deferred()

    def log(self, *values: object, sep: str = " ", level: int = logging.INFO) -> None:
        """Append a level-tagged message to the panel, or print it without a panel.

        Args:
            *values: Objects joined like ``print``.
            sep: Separator between values.
            level: A ``logging`` level, such as ``logging.WARNING``.

        Raises:
            TypeError: If ``level`` is not an int, or is a bool.

        Example:
            >>> quiet = Options(disable=True, stream=sys.stdout)
            >>> with Bar(total=2, options=quiet) as bar:
            ...     bar.log("loss spiked to", 4.2, level=logging.WARNING)
            warn  loss spiked to 4.2
        """
        if not isinstance(level, int) or isinstance(level, bool):
            raise TypeError(f"level must be an int, not {type(level).__name__}")
        stamp = log_stamp(time.time()) if self._options.log_time else ""
        lines = log_lines(_lines(values, sep), level, stamp)
        self._show(lambda engine: engine.log(lines))

    def panel(self, content: object) -> None:
        """Replace the panel's content (None or empty clears it).

        Records (mappings, dataclasses, named tuples) show keys beside values, and a
        sequence of them or a DataFrame a table; other sequences a row per item.

        Args:
            content: New panel contents.

        Example:
            >>> with Bar(total=2, options=Options(disable=True)) as bar:
            ...     bar.panel({"loss": 0.1376, "lr": 3e-4})
            ...     bar.panel([{"worker": 1, "state": "busy"}])
        """
        layout = partial(panel_lines, content, justify=self._options.panel_justify)
        self._show(lambda engine: engine.set_panel(layout))

    def write(self, *values: object, sep: str = " ") -> None:
        """Print text permanently above the live region, after any logged lines.

        Args:
            *values: Objects joined like ``print``.
            sep: Separator between values.
        """
        lines = _lines(values, sep)
        self._show(lambda engine: engine.write(lines))

    def reset(self, total: float | None = None) -> None:
        """Restart counting and timing from zero, with a new ``total`` unless None.

        Args:
            total: The new expected count, or None to keep the current one.

        Raises:
            TypeError: If ``total`` is not a number.
            ValueError: If ``total`` is negative or not finite.
            RuntimeError: If the bar is closed.
        """
        validated = _validate_total(total)
        with LOCK:
            if self._outcome is not None:
                raise RuntimeError("a closed bar cannot be reset")
            self._initial -= self._count()
            if validated is not None:
                self._total, self._sized = validated, None
            self._start(time.monotonic())

    def close(self) -> None:
        """Finish the bar; idempotent."""
        self._finish(None)

    def interrupt(self) -> None:
        """Finish the bar as interrupted; idempotent."""
        self._finish("interrupted")

    def _abandon(self) -> None:
        """Finish the bar at interpreter exit; a manual bar without a total is kept."""
        self._finish(None, strict=True)

    def _finish(
        self, outcome: Outcome | None, cause: str = "", strict: bool = False
    ) -> None:
        if self._engine is not None and self._outcome is None:
            _engine.flush_stdout(self._engine.stream)
        with LOCK:
            final = self._conclude(outcome, cause, strict)
        if final is not None:
            run_deferred()

    def _conclude(
        self, outcome: Outcome | None, cause: str = "", strict: bool = False
    ) -> Frame | None:
        """Decide the outcome and ``cause``, unregister, and return the last frame."""
        if self._outcome is not None:
            return None
        count, total = self._count(), self._total
        if self._wraps_iterable:
            complete = self._exhausted
        else:
            complete = (not strict) if total is None else count >= total
        self._outcome = outcome or ("done" if complete else "interrupted")
        self._cause = cause
        try:
            final = self._sample(time.monotonic())
        except BaseException:
            self._outcome = None
            raise
        self._sized = None
        self._fetched = repeat(self._fetched()).__next__
        self._push = _ignore
        if self._engine is not None:
            self._engine.remove(self, final)
        return final

    def __enter__(self) -> Self:
        """Return this bar for a ``with`` block, which closes it."""
        self._managed = True
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *exc_info: object) -> None:
        """Close the bar: failed on exceptions (its cause shown), else interrupted."""
        self._finish(*ending(exc_type))

    def __bool__(self) -> bool:
        """Return True, so a bar without a known total is still truthy."""
        return True

    def __repr__(self) -> str:
        """Return the class name, description, count, and total."""
        name = type(self).__name__
        return f"{name}(desc={self._desc!r}, n={self.n}, total={self.total})"


@wraps(Bar.update)
def _update_per_thread(self: Bar, n: float = 1) -> None:
    push = self._push
    if push is _thread_queue:
        try:
            push = _THREAD.pushes[self]
        except (AttributeError, KeyError):
            push = self._thread_push()
    push(n)


if not _ONE_QUEUE:
    Bar.update = _update_per_thread  # type: ignore[method-assign]


def _accumulate(manual: float, increments: Iterator[float], base: float) -> float:
    """Add increments to ``manual`` as they are drained, summing in C."""
    try:
        batch = sum(increments)
        updated = manual + (batch if type(batch) is int else float(batch))
        if math.isfinite(base + updated):
            return updated
        reason = "the total is not finite"
    except Exception as error:
        deque(increments, maxlen=0)  # pop the rest of the discarded batch
        reason = describe(error)
    message = f"psnek discarded a batch of updates with an invalid value: {reason}"
    once(logging.ERROR, message)
    return manual


def _validate_total(total: float | None) -> float | None:
    if total is None:
        return None
    try:
        if isinstance(total, _NOT_NUMBERS):
            raise TypeError
        value = float(total)
    except TypeError as error:
        raise TypeError(f"total must be numeric, not {type(total).__name__}") from error
    except (ValueError, OverflowError) as error:
        raise ValueError(f"total must be a finite number, got {total!r}") from error
    if not 0 <= value < math.inf:
        raise ValueError(f"total must be a finite non-negative number, got {total!r}")
    return total if type(total) is int else value + 0.0


def _apply(
    function: Callable[[T], R], items: Sequence[T], start: int, stop: int
) -> list[R]:
    return list(map(function, items[start:stop]))


def _advance(
    bar: Bar,
    credited: "set[Future[list[R]]]",
    lock: threading.Lock,
    future: "Future[list[R]]",
) -> None:
    """Count a finished chunk once, from its callback or from the caller."""
    with lock:
        if future in credited:
            return
        credited.add(future)
        if not future.cancelled() and future.exception() is None:
            bar.update(len(future.result()))


def _raised(future: "Future[list[R]]") -> bool:
    return future.done() and not future.cancelled() and future.exception() is not None


def pmap(
    function: Callable[[T], R],
    iterable: Iterable[T],
    desc: object = "",
    *,
    workers: int | Executor | None = None,
    options: Options | None = None,
) -> list[R]:
    """Apply ``function`` to every item in parallel, in chunks, preserving order.

    Args:
        function: Callable applied to each item.
        iterable: Input items.
        desc: Short description shown before the bar.
        workers: Thread count for a new pool, or an executor (such as a process pool).
        options: Presentation settings for the bar.

    Returns:
        Results in input order.

    Raises:
        TypeError: If ``workers`` or ``options`` has the wrong type.
        ValueError: If ``workers`` is an int less than 1.
        Exception: The first failure of iterating or of ``function``, in input order.

    Example:
        >>> pmap(abs, [-1, 2, -3], workers=2, options=Options(disable=True))
        [1, 2, 3]
    """
    if not isinstance(workers, int | Executor | None) or isinstance(workers, bool):
        raise TypeError(f"workers must be an int or an Executor, not {workers!r}")
    if isinstance(workers, int) and workers < 1:
        raise ValueError("workers must be at least 1")
    items = iterable if isinstance(iterable, _SLICEABLE) else list(iterable)
    total = len(items)
    pool = workers if isinstance(workers, Executor) else ThreadPoolExecutor(workers)
    count = getattr(pool, "_max_workers", None) or os.cpu_count() or 1
    size = max(total // (CHUNKS_PER_WORKER * count), 1)
    copy = not isinstance(pool, ThreadPoolExecutor)
    futures: list[Future[list[R]]] = []
    credited: set[Future[list[R]]] = set()
    lock = threading.Lock()
    try:
        with Bar(total, desc, options=options) as bar:
            for start in range(0, total, size):
                stop = min(start + size, total)
                chunk, begin = (items[start:stop], 0) if copy else (items, start)
                future = pool.submit(
                    _apply, function, chunk, begin, begin + stop - start
                )
                future.add_done_callback(partial(_advance, bar, credited, lock))
                futures.append(future)
                if _raised(future):
                    break
            wait(futures, return_when=FIRST_EXCEPTION)
            failure = next((job.exception() for job in futures if _raised(job)), None)
            if failure is not None:
                raise failure
            for future in futures:
                _advance(bar, credited, lock, future)
            return list(chain.from_iterable(future.result() for future in futures))
    finally:
        for future in futures:
            future.cancel()
        if pool is not workers:
            pool.shutdown(wait=False, cancel_futures=True)


def write(*values: object, sep: str = " ") -> None:
    """Print text permanently above the active bars, like ``tqdm.write``.

    Args:
        *values: Objects joined like ``print``.
        sep: Separator between values.
    """
    _shown_engine().write(_lines(values, sep))
    run_deferred()


class LogHandler(logging.Handler):
    """Logging handler writing records, tagged with their level, into the panel.

    Example:
        >>> handler = LogHandler(log_time=True)
    """

    def __init__(
        self, level: int | str = logging.NOTSET, *, log_time: bool = False
    ) -> None:
        """Create a handler for records at ``level`` or above.

        Args:
            level: The lowest ``logging`` level handled, or its name.
            log_time: Start each record with its local wall-clock time.
        """
        super().__init__(level)
        self.log_time = log_time

    def emit(self, record: logging.LogRecord) -> None:
        """Format a record and route it to the panel, reporting failures."""
        try:
            stamp = log_stamp(record.created) if self.log_time else ""
            lines = log_lines(self.format(record).split("\n"), record.levelno, stamp)
            _shown_engine().log(lines)
        except Exception:
            self.handleError(record)
