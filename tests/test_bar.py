"""Tests for counting semantics, lifecycle, threading, and async iteration."""

import asyncio
import contextlib
import enum
import gc
import io
import logging
import threading
import time
import weakref
from collections import deque
from collections.abc import AsyncIterator, Callable, Generator, Iterable, Iterator
from concurrent import futures
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from decimal import Decimal
from typing import NamedTuple, ParamSpec, TextIO, TypeVar, cast

import pytest

import psnek
from psnek import (
    AsyncTrack,
    Bar,
    LogHandler,
    Options,
    Track,
    _bar,
    _engine,
    atrack,
    pmap,
    track,
)
from psnek._bar import PanelPosition
from psnek._table import Justify
from psnek._text import Scale
from tests.fakes import (
    QUIET,
    FakeTerminal,
    UnprintableError,
    engine_lock,
    take_steps,
)


def _generate(count: int) -> Iterator[int]:
    yield from range(count)


def _fail_at(item: int) -> int:
    if item == 3:
        raise ValueError("bad item")
    return item


class _Count:
    def __init__(self, value: int) -> None:
        self.value = value

    def __index__(self) -> int:
        return self.value


def test_n_counts_completed_items_once() -> None:
    # Arrange
    sources: list[Iterable[int]] = [
        list(range(5)),
        _generate(5),
        {i: i for i in range(5)},
    ]

    for source in sources:
        bar = track(source, total=5, options=QUIET)

        # Act
        seen = [bar.n for _ in bar]

        # Assert
        assert seen == [0, 1, 2, 3, 4]
        assert (bar.n, bar.outcome) == (5, "done")
        with pytest.raises(RuntimeError):
            iter(bar)


def test_worklist_totals_grow_with_the_list_and_release_it() -> None:
    # Arrange
    queue = [0]
    bar = track(queue, options=QUIET)
    totals: list[float | None] = []

    # Act
    for item in bar:
        queue.extend([item + 1] if item < 4 else [])
        totals.append(bar.total)
        with engine_lock():
            frame = bar._sample(time.monotonic())
        assert frame.total == len(queue)

    # Assert
    assert totals == [2, 3, 4, 5, 5]
    assert (bar.n, bar.total, bar.outcome) == (5, 5, "done")
    assert bar not in gc.get_referrers(queue)


class _LazyLength:
    def __len__(self) -> int:
        raise TypeError("length unknown")

    def __iter__(self) -> Iterator[int]:
        return iter(range(3))


class _UnsizedAfterAll(list[int]):
    def __len__(self) -> int:
        raise NotImplementedError("length is lazy")


def test_track_infers_totals() -> None:
    # Arrange
    items = deque([1, 2])
    from_deque = track(items, options=QUIET)
    items.append(3)
    huge = track(range(2**64), options=QUIET)
    oversized = track(range(3), total=2**70, options=QUIET)
    from_int = track(3, options=QUIET)

    # Act
    for item in huge:
        if item == 3:
            break

    # Assert
    assert from_deque.total == 2
    assert (huge.total, huge.n) == (None, 3)
    assert list(oversized) == [0, 1, 2]
    assert oversized.total == 2**70
    with pytest.raises(TypeError, match="no total that fits"):
        len(oversized)
    assert list(from_int) == [0, 1, 2]
    assert (from_int.total, len(from_int)) == (3, 3)
    assert list(track(_Count(3), options=QUIET)) == [0, 1, 2]
    for unsized in (_LazyLength(), _UnsizedAfterAll([0, 1, 2])):
        bar = track(unsized, options=QUIET)
        assert list(bar) == [0, 1, 2]
        assert bar.total is None


def test_nested_tracks_share_the_total() -> None:
    # Arrange
    unsized = track(_generate(4), options=QUIET)
    sized = track(range(4), options=QUIET)

    # Act
    unsized_outer = track(unsized, options=QUIET)
    sized_outer = track(sized, options=QUIET)

    # Assert
    assert bool(unsized)
    with pytest.raises(TypeError):
        len(unsized)
    assert (unsized_outer.total, list(unsized_outer)) == (None, [0, 1, 2, 3])
    assert (sized_outer.total, list(sized_outer)) == (4, [0, 1, 2, 3])


@pytest.mark.parametrize(
    ("make", "total", "items"),
    [
        (lambda: psnek.zip("abc", [1, 2], options=QUIET), 2, [("a", 1), ("b", 2)]),
        (
            lambda: psnek.zip(_generate(2), "abc", options=QUIET),
            3,
            [(0, "a"), (1, "b")],
        ),
        (lambda: psnek.zip(_generate(2), options=QUIET), None, [(0,), (1,)]),
        (lambda: psnek.zip(_generate(2), total=2, options=QUIET), 2, [(0,), (1,)]),
        (lambda: psnek.zip(options=QUIET), None, []),
    ],
)
def test_zip_totals_its_shortest_sized_source(
    make: Callable[[], Track[tuple[object, ...]]],
    total: int | None,
    items: list[tuple[object, ...]],
) -> None:
    # Act
    bar = make()
    yielded = list(bar)

    # Assert
    assert (bar.total, yielded) == (total, items)
    assert bar.outcome == "done"


def test_builtin_enumerate_over_a_track_keeps_its_total() -> None:
    # Arrange
    bar = track("ab", options=QUIET)

    # Act
    numbered = list(enumerate(bar, start=1))

    # Assert
    assert (bar.total, numbered) == (2, [(1, "a"), (2, "b")])
    assert bar.outcome == "done"


def test_strict_zip_raises_on_a_length_mismatch() -> None:
    # Arrange
    bar = psnek.zip("ab", "abc", desc="pairs", strict=True, options=QUIET)

    # Act / Assert
    with pytest.raises(ValueError, match="longer"):
        list(bar)


def test_zip_starts_a_wrapped_bar_only_when_iterated() -> None:
    # Arrange
    log = io.StringIO()
    options = Options(stream=log, leave=True)
    inner = track([1, 2], "inner", options=options)
    outer = psnek.zip(inner, "ab", desc="outer", options=options)

    # Act
    items = list(outer)

    # Assert
    started = [line.split(":")[0] for line in log.getvalue().splitlines()[:2]]
    assert started == ["outer", "inner"]
    assert len(items) == 2
    assert inner.outcome == outer.outcome == "done"


def test_zip_refuses_non_iterables_when_built() -> None:
    # Act / Assert
    with pytest.raises(TypeError, match="expected iterables, got int"):
        psnek.zip("ab", 3, options=QUIET)  # type: ignore[call-overload]


class _Letters:
    """An iterable through the old ``__getitem__`` protocol only."""

    def __getitem__(self, position: int) -> str:
        return "abcd"[position]


class _IndexLike:
    """An index-like value that also has ``__getitem__``, as NumPy integers do."""

    def __index__(self) -> int:
        return 3

    def __getitem__(self, key: object) -> int:
        raise TypeError("not a sequence")


def test_sequence_protocol_iterables_track_and_index_likes_stay_counts() -> None:
    # Arrange
    letters = cast("Iterable[str]", _Letters())

    # Act
    tracked = list(track(letters, options=QUIET))
    paired = list(psnek.zip(letters, "xy", options=QUIET))
    counted = list(track(_IndexLike(), options=QUIET))

    # Assert
    assert tracked == ["a", "b", "c", "d"]
    assert paired == [("a", "x"), ("b", "y")]
    assert counted == [0, 1, 2]
    with pytest.raises(TypeError, match="iterable or int, got float"):
        track(1.5, options=QUIET)  # type: ignore[call-overload]


def test_track_is_displayed_only_when_iterated(terminal: FakeTerminal) -> None:
    # Arrange
    bar = track(range(3), options=Options(stream=terminal))

    # Act
    registered_before = bool(_engine.ENGINES)
    list(bar)

    # Assert
    assert not registered_before
    assert bar.outcome == "done"


class _Marker:
    """An object that can be weakly referenced."""


def test_closing_early_is_interrupted_and_releases_the_source() -> None:
    # Arrange
    marker = _Marker()
    alive = weakref.ref(marker)
    broken = track((0, marker, 2), options=QUIET)
    del marker
    unbounded = Bar(options=QUIET)
    short = Bar(total=5, options=QUIET)

    # Act
    for _ in broken:
        break
    gc.collect()
    unbounded.update(3)
    unbounded.close()
    short.update(2)
    short.close()

    # Assert
    assert alive() is None
    assert (broken.outcome, broken.n) == ("interrupted", 0)
    assert unbounded.outcome == "done"
    assert short.outcome == "interrupted"


def test_errors_mark_the_bar_failed() -> None:
    # Arrange
    def failing() -> Iterator[int]:
        yield from range(3)
        raise RuntimeError("source broke")

    source_bar = track(failing(), options=QUIET)

    # Act
    with (
        pytest.raises(ValueError, match="bad"),
        track(_generate(10), options=QUIET) as body_bar,
    ):
        for item in body_bar:
            _fail_at(item)
    with pytest.raises(ValueError, match="after"), track([1, 2], options=QUIET) as bar:
        for _ in bar:
            pass
        open_after_loop = bar.outcome is None
        raise ValueError("after the loop")
    with pytest.raises(RuntimeError, match="source broke"):
        for _ in source_bar:
            pass

    # Assert
    assert body_bar.outcome == "failed"
    assert open_after_loop
    assert bar.outcome == "failed"
    assert source_bar.n == 2


def test_cancellation_marks_interrupted() -> None:
    for error in (KeyboardInterrupt, SystemExit):
        # Act
        with pytest.raises(error), Bar(total=5, options=QUIET) as bar:
            raise error

        # Assert
        assert bar.outcome == "interrupted"


class _Level(enum.IntEnum):
    """Custom logging levels, as some applications define them."""

    NOTICE = 25


class _Halt(BaseException):
    """An interruption psnek has no friendlier name for."""


def test_ending_names_the_outcome_and_cause() -> None:
    # Arrange
    errors: dict[type[BaseException] | None, tuple[str | None, str]] = {
        None: (None, ""),
        KeyboardInterrupt: ("interrupted", "Ctrl-C"),
        SystemExit: ("interrupted", "exit"),
        GeneratorExit: ("interrupted", "closed"),
        asyncio.CancelledError: ("interrupted", "cancelled"),
        futures.CancelledError: ("failed", "CancelledError"),
        _Halt: ("interrupted", "_Halt"),
        ValueError: ("failed", "ValueError"),
        MemoryError: ("failed", "MemoryError"),
    }

    # Act
    endings = {error: _bar.ending(error) for error in errors}

    # Assert
    assert endings == errors


@pytest.fixture
def log_stream(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """A stream that receives each bar's plain log-mode lines, its final one too."""
    monkeypatch.setenv("PSNEK_MODE", "log")
    return io.StringIO()


def test_an_exception_leaving_a_bar_is_shown_as_its_cause_beside_its_state(
    log_stream: io.StringIO,
) -> None:
    # Arrange
    options = Options(stream=log_stream, leave=True)

    # Act
    with pytest.raises(KeyboardInterrupt), Bar(10, "train", options=options) as bar:
        bar.status(loss=0.5)
        bar.update(3)
        raise KeyboardInterrupt
    with pytest.raises(ValueError, match="bad"), Bar(10, options=options) as failed:
        raise ValueError("bad")
    with Bar(1, options=options) as done:
        done.update()

    def produce() -> Generator[int, None, None]:
        with Bar(5, options=options) as bar:
            for item in range(5):
                yield item
                bar.update()

    generator = produce()
    next(generator)
    generator.close()

    # Assert
    lines = log_stream.getvalue().splitlines()
    interrupted, failure, finished = lines[1:-2:2]
    assert " interrupted Ctrl-C " in interrupted
    assert interrupted.endswith("loss=0.5000")
    assert " failed ValueError " in failure
    assert " done " in finished
    assert finished.endswith("it/s")
    assert failed.outcome == "failed"
    assert " interrupted closed " in lines[-1]


@pytest.mark.parametrize("gradient", [True, False])
def test_the_gradient_option_sets_whether_bars_are_colored(
    terminal: FakeTerminal, gradient: bool
) -> None:
    # Act
    options = Options(stream=terminal, gradient=gradient)
    for _ in track(range(3), "resize", options=options):
        pass

    # Assert
    assert ("\x1b[38;" in terminal.getvalue()) is gradient


def test_a_gradient_of_hex_colors_colors_the_bar(terminal: FakeTerminal) -> None:
    # Arrange
    options = Options(stream=terminal, gradient=["#FF0000", "0000ff"])

    # Act
    for _ in track(range(3), "resize", options=options):
        pass

    # Assert
    assert options.gradient == ("#FF0000", "0000ff")
    assert hash(options) == hash(Options(stream=terminal, gradient=options.gradient))
    output = terminal.getvalue()
    assert "\x1b[1;38;5;21m" in output
    assert "\x1b[38;5;196m" in output


def test_invalid_arguments_are_rejected() -> None:
    # Arrange
    cases: list[tuple[Callable[[], object], type[Exception]]] = [
        (lambda: track(cast("Iterable[int]", 2.5), options=QUIET), TypeError),
        (lambda: Bar(total=-1, options=QUIET), ValueError),
        (lambda: Bar(total=float("nan"), options=QUIET), ValueError),
        (lambda: Bar(total=10**400, options=QUIET), ValueError),
        (lambda: Bar(total=cast("float", "10"), options=QUIET), TypeError),
        (lambda: Bar(total=True, options=QUIET), TypeError),
        (lambda: Options(scale=cast("Scale", "SI")), ValueError),
        (lambda: Options(panel_position=cast("PanelPosition", "up")), ValueError),
        (lambda: Options(panel_justify=cast("Justify", "justified")), ValueError),
        (lambda: Options(panel_lines=-1), ValueError),
        (lambda: Options(panel_lines=True), TypeError),
        (lambda: Options(height=0), ValueError),
        (lambda: Options(height=True), TypeError),
        (lambda: Options(height=cast("int", 2.0)), TypeError),
        (lambda: Options(unit=cast("str", None)), TypeError),
        (lambda: Options(leave=cast("bool", 1)), TypeError),
        (lambda: Options(notify=cast("bool", "no")), TypeError),
        (lambda: Options(log_time=cast("bool", 1)), TypeError),
        (lambda: Options(gradient=cast("bool", "no")), TypeError),
        (lambda: Options(gradient=cast("bool", "#ff0000")), TypeError),
        (lambda: Options(gradient=cast("list[str]", [0xFF0000])), TypeError),
        (lambda: Options(gradient=[]), ValueError),
        (lambda: Options(gradient=["#ff0000", "red"]), ValueError),
        (lambda: Options(stream=cast("TextIO", "stderr")), TypeError),
        (lambda: Bar(options=cast("Options", {"disable": True})), TypeError),
        (lambda: atrack(cast("AsyncIterator[int]", [1]), options=QUIET), TypeError),
        (lambda: pmap(abs, [1], workers=0, options=QUIET), ValueError),
        (lambda: pmap(abs, [1], workers=cast("int", "4"), options=QUIET), TypeError),
        (lambda: pmap(abs, cast("list[int]", 5), options=QUIET), TypeError),
    ]

    for make, error in cases:
        # Act / Assert
        with pytest.raises(error):
            make()


def test_totals_and_descriptions_are_normalized_and_settable() -> None:
    # Arrange
    with Bar(total=2, desc="old", options=QUIET) as bar:
        # Act
        bar.total = 4
        bar.desc = cast("str", 42)
        bar.update(4)
    zero = Bar(total=-0.0, desc=cast("str", None), options=QUIET)
    decimal = Bar(total=cast("float", Decimal("2.5")), options=QUIET)

    # Assert
    assert (bar.total, bar.desc, bar.outcome) == (4, "42", "done")
    assert type(bar.total) is int
    with pytest.raises(ValueError, match="total"):
        bar.total = -1
    assert (str(zero.total), zero.desc) == ("0.0", "None")
    assert decimal.total == 2.5
    zero.close()
    decimal.close()


def test_concurrent_updates_are_never_lost() -> None:
    # Arrange
    bar = Bar(total=80_000, options=QUIET)

    def work() -> None:
        for _ in range(10_000):
            bar.update()

    threads = [threading.Thread(target=work) for _ in range(8)]

    # Act
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    bar.close()

    # Assert
    assert bar.n == 80_000
    assert bar.outcome == "done"


@pytest.fixture
def free_threaded(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make new bars behave as on a free-threaded build with the GIL disabled."""
    monkeypatch.setattr(_bar, "_ONE_QUEUE", False)
    monkeypatch.setattr(Bar, "update", _bar._update_per_thread)


def test_update_uses_one_queue_only_while_the_gil_is_enabled() -> None:
    # Act
    per_thread = Bar.update is _bar._update_per_thread

    # Assert
    assert per_thread is not _bar._gil_enabled()
    assert Bar.update.__doc__ is not None
    assert Bar.update.__doc__.startswith("Record ``n`` more completed units.")


@pytest.mark.usefixtures("free_threaded")
def test_without_the_gil_each_updating_thread_gets_its_own_queue() -> None:
    # Arrange
    bar = Bar(total=80_004, options=QUIET)

    def work() -> None:
        for _ in range(10_000):
            bar.update()

    threads = [threading.Thread(target=work) for _ in range(8)]

    # Act
    bar.update(2)
    bar.update(2)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    queues = len(bar._queues)
    bar.close()
    bar.update()

    # Assert
    assert queues == 9
    assert bar.n == 80_004
    assert bar.outcome == "done"
    assert not any(bar._queues)


@pytest.mark.usefixtures("free_threaded")
def test_without_the_gil_threads_drop_their_queues_of_closed_bars() -> None:
    # Arrange
    with Bar(options=QUIET) as first:
        first.update()
    second = Bar(options=QUIET)

    # Act
    second.update()

    # Assert
    assert list(_bar._THREAD.pushes) == [second]
    second.close()


def test_increments_of_any_numeric_type_are_validated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    cases: list[tuple[list[float], float]] = [
        ([cast("float", Decimal("1.5")), cast("float", Decimal("0.5"))], 2.0),
        ([2, -1], 1),
        ([cast("float", 1j)], 0),
        ([10**400], 0),
        ([1e308, 1e308], 0),
    ]

    for increments, expected in cases:
        caplog.clear()
        bar = Bar(options=QUIET)

        # Act
        for increment in increments:
            bar.update(increment)
        count = bar.n
        bar.close()

        # Assert
        assert count == expected
        assert type(count) in (int, float)
        assert ("psnek discarded a batch" in caplog.text) is (expected == 0)


def test_a_batch_with_a_bad_value_is_discarded_whole_and_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    bar = Bar(options=QUIET)

    # Act
    bar.update(cast("float", "x"))
    bar.update(1)
    bar.update(1)
    discarded = bar.n
    bar.update(1)
    counted = bar.n
    bar.update(cast("float", "x"))
    again = bar.n
    bar.close()

    # Assert
    assert (discarded, counted, again) == (0, 1, 1)
    assert caplog.text.count("psnek discarded a batch") == 1


class _Exploding:
    def __radd__(self, other: object) -> float:
        raise RuntimeError("cannot add")


class _UnprintableSum:
    def __radd__(self, other: object) -> float:
        raise UnprintableError


class _UnprintableLength:
    def __iter__(self) -> Iterator[int]:
        return iter(range(2))

    def __len__(self) -> int:
        raise UnprintableError


def test_any_error_from_user_values_is_contained_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    bar = Bar(total=3, options=QUIET)
    bar.update(cast("float", _Exploding()))
    exploded = bar.n
    bar.update(cast("float", _UnprintableSum()))

    # Act
    unprintable = bar.n
    bar.update(3)
    bar.close()
    with caplog.at_level(logging.DEBUG, logger="psnek"):
        unsized = track(_UnprintableLength(), options=QUIET)

    # Assert
    assert (exploded, unprintable, bar.outcome) == (0, 0, "done")
    assert unsized.total is None
    assert "RuntimeError: cannot add" in caplog.text
    assert "invalid value: UnprintableError" in caplog.text
    assert "len() failed: UnprintableError" in caplog.text


def test_closed_bars_ignore_updates_and_refuse_reset() -> None:
    # Arrange
    bar = Bar(total=2, options=QUIET)
    bar.update(2)
    bar.close()
    elapsed = bar.elapsed

    # Act
    bar.update(5)

    # Assert
    with pytest.raises(RuntimeError, match="closed"):
        bar.reset(total=5)
    assert (bar.n, bar.total, bar.elapsed) == (2, 2, elapsed)


def test_hidden_bars_are_drained_by_a_render_thread() -> None:
    # Arrange
    bar = Bar(options=QUIET)

    # Act
    bar.update(3)
    for _ in range(200):
        if not any(bar._queues):
            break
        time.sleep(_engine.TICK / 10)

    # Assert
    assert bar._engine is not None
    assert bar._engine.hidden
    assert not any(bar._queues)
    assert bar.n == 3
    bar.close()


def test_fractional_updates_and_reset() -> None:
    # Arrange
    with Bar(total=2, options=QUIET) as bar:
        bar.update(0.5)
        bar.update(0.25)
        before = bar.n
        with pytest.raises(ValueError, match="total"):
            bar.reset(total=-1)
        rejected = (bar.n, bar.total)

        # Act
        bar.reset(total=4)
        bar.update(4)

    # Assert
    assert (before, rejected) == (0.75, (0.75, 2))
    assert (bar.total, bar.n) == (4, 4)


def test_status_merges_formats_and_removes() -> None:
    # Arrange
    with Bar(options=QUIET) as bar:
        # Act
        bar.status(loss=0.123456, step=3, tag="a")
        bar.status(step=None)
        with engine_lock():
            frame = bar._sample(0.0)

    # Assert
    assert frame.status == (("loss", "0.1235"), ("tag", "a"))
    assert "Bar(desc=''" in repr(bar)


@dataclass
class _Metrics:
    loss: float
    note: str = field(default="hidden", repr=False)


class _Step(NamedTuple):
    epoch: int
    lr: float


def test_status_takes_a_mapping_dataclass_or_named_tuple_like_the_panel() -> None:
    # Arrange
    with Bar(options=QUIET) as bar:
        # Act
        bar.status({"val loss": 0.21, 3: "x", "gone": None}, lr=3e-4)
        with engine_lock():
            mapped = bar._sample(0.0).status
        bar.status(_Metrics(0.5))
        bar.status(_Step(2, 1e-3), lr=None, epoch=3)
        with engine_lock():
            merged = bar._sample(0.0).status

    # Assert
    assert mapped == (("val loss", "0.2100"), ("3", "x"), ("lr", "0.0003000"))
    assert merged == (
        ("val loss", "0.2100"),
        ("3", "x"),
        ("loss", "0.5000"),
        ("epoch", "3"),
    )


@pytest.mark.parametrize("values", [[("loss", 0.1)], "loss=0.1", 0.1])
def test_status_rejects_values_that_are_not_records(values: object) -> None:
    with Bar(options=QUIET) as bar, pytest.raises(TypeError, match="mapping"):
        bar.status(values)


def test_stall_detection_waits_for_warm_up_then_escalates() -> None:
    # Arrange
    with Bar(total=10, options=QUIET) as bar, engine_lock():
        start = bar._pace.changed_at
        bar.update()
        bar._tally(start + 0.5)
        warming = [bar._sample(start + 0.5 + offset).state for offset in (6, 600)]
    with Bar(total=10, options=QUIET) as bar, engine_lock():
        last = take_steps(bar, bar._pace.changed_at)

        # Act
        states = [bar._sample(last + offset).state for offset in (1, 6, 31)]
        bar.update()
        recovered = bar._sample(last + 32).state
    with Bar(total=10, options=QUIET) as hidden, engine_lock():
        last = take_steps(hidden, hidden._pace.changed_at)
        drained = hidden._sample(last + 40)

    # Assert
    assert warming == ["running", "running"]
    assert states == ["running", "stalled", "stuck"]
    assert recovered == "running"
    assert (drained.state, drained.idle) == ("stuck", 40)


def test_the_snake_slows_down_while_stalled_and_further_while_stuck() -> None:
    # Arrange
    with Bar(options=QUIET) as bar, engine_lock():
        last = take_steps(bar, bar._pace.changed_at)

        # Act
        speeds = {}
        for offset in (1, 6, 31):
            before = bar._sample(last + offset)
            after = bar._sample(last + offset + 0.5)
            speeds[after.state] = (after.phase - before.phase) / 0.5

    # Assert
    assert list(speeds) == ["running", "stalled", "stuck"]
    assert speeds["running"] > speeds["stalled"] > speeds["stuck"] > 0


def test_stall_threshold_widens_with_the_spread_of_gaps() -> None:
    # Arrange
    with Bar(total=10, options=QUIET) as steady, engine_lock():
        steady_last = take_steps(steady, steady._pace.changed_at, (5.0,) * 4)

        # Act
        steady_state = steady._sample(steady_last + 21).state
    with Bar(total=10, options=QUIET) as bursty, engine_lock():
        bursty_last = take_steps(bursty, bursty._pace.changed_at, (0.5, 0.5, 20, 0.5))
        bursty_state = bursty._sample(bursty_last + 21).state

    # Assert
    assert bursty._pace._gap_mean < steady._pace._gap_mean
    assert (steady_state, bursty_state) == ("stalled", "running")


async def _produce(count: int) -> AsyncIterator[int]:
    for item in range(count):
        await asyncio.sleep(0)
        yield item


def test_async_track_counts_after_each_body_and_closes() -> None:
    # Arrange
    async def consume() -> tuple[list[float], float, str | None]:
        bar = atrack(_produce(4), total=4, options=QUIET)
        counts = [bar.n async for _ in bar]
        return counts, bar.n, bar.outcome

    # Act
    counts, count, outcome = asyncio.run(consume())

    # Assert
    assert counts == [0, 1, 2, 3]
    assert count == 4
    assert outcome == "done"


def test_async_bars_need_async_with_and_stay_open_inside_it() -> None:
    # Arrange
    async def consume() -> tuple[str | None, str | None]:
        async with atrack(_produce(2), options=QUIET) as bar:
            async for _ in bar:
                pass
            inside = bar.outcome
        return inside, bar.outcome

    plain = atrack(_produce(1), options=QUIET)

    # Act
    inside, after = asyncio.run(consume())

    # Assert
    assert (inside, after) == (None, "done")
    with pytest.raises(TypeError, match="async with"), plain:
        pass
    plain.close()


class _BrokenAsyncIterable:
    def __aiter__(self) -> AsyncIterator[int]:
        raise TypeError("not today")


def test_async_source_errors_mark_the_bar_failed() -> None:
    # Arrange
    async def broken() -> AsyncIterator[int]:
        yield 1
        raise ValueError("source broke")

    sources: list[tuple[AsyncIterator[int], type[Exception], str]] = [
        (broken(), ValueError, "source broke"),
        (cast("AsyncIterator[int]", _BrokenAsyncIterable()), TypeError, "not today"),
    ]

    for source, error, message in sources:
        bar = atrack(source, options=QUIET)

        async def consume(bar: AsyncTrack[int] = bar) -> None:
            async for _ in bar:
                pass

        # Act
        with pytest.raises(error, match=message):
            asyncio.run(consume())

        # Assert
        assert bar.outcome == "failed"
        assert bar._cause == error.__name__


class _BrokenIterable:
    def __iter__(self) -> Iterator[int]:
        raise OSError("disk gone")


def test_an_iterable_that_fails_to_start_marks_the_bar_failed() -> None:
    # Arrange
    bar = track(_BrokenIterable(), options=QUIET)

    # Act
    with pytest.raises(OSError, match="disk gone"):
        list(bar)

    # Assert
    assert bar.outcome == "failed"
    assert bar._cause == "OSError"


def test_cancelled_async_bars_are_interrupted() -> None:
    # Arrange
    async def endless() -> AsyncIterator[int]:
        while True:
            yield 1

    async def consume(bar: AsyncTrack[int]) -> None:
        async with bar:
            async for _ in bar:
                await asyncio.sleep(0)

    async def run() -> str | None:
        bar = atrack(endless(), options=QUIET)
        task = asyncio.ensure_future(consume(bar))
        await asyncio.sleep(0.01)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return bar.outcome

    # Act
    outcome = asyncio.run(run())

    # Assert
    assert outcome == "interrupted"


class _Countdown:
    """An async iterator that can be weakly referenced."""

    def __init__(self) -> None:
        self.left = 3

    def __aiter__(self) -> "_Countdown":
        return self

    async def __anext__(self) -> int:
        self.left -= 1
        if self.left < 0:
            raise StopAsyncIteration
        return self.left


def test_breaking_an_async_loop_closes_the_bar_and_releases_the_source() -> None:
    # Arrange
    source = _Countdown()
    alive = weakref.ref(source)
    bar = atrack(source, options=QUIET)
    del source

    async def consume() -> None:
        async for _ in bar:
            break

    # Act
    asyncio.run(consume())

    # Assert
    assert alive() is None
    assert bar.outcome == "interrupted"


def test_pmap_preserves_order_with_threads() -> None:
    # Act
    squares = pmap(lambda value: value * value, range(1000), workers=4, options=QUIET)
    from_deque = pmap(abs, deque([-1, 2, -3]), workers=2, options=QUIET)

    # Assert
    assert squares == [value * value for value in range(1000)]
    assert from_deque == [1, 2, 3]


@pytest.fixture
def chunk_sizes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the size of every chunk pmap sends, still applying the function."""
    sizes: list[int] = []

    def record(
        fn: Callable[[int], int], items: list[int], start: int, stop: int
    ) -> list[int]:
        sizes.append(stop - start)
        return [fn(items[index]) for index in range(start, stop)]

    monkeypatch.setattr("psnek._bar._apply", record)
    return sizes


def test_pmap_sizes_chunks_by_the_worker_count(
    monkeypatch: pytest.MonkeyPatch, chunk_sizes: list[int]
) -> None:
    # Arrange
    pmap(abs, list(range(160)), workers=2, options=QUIET)
    by_count = list(chunk_sizes)
    chunk_sizes.clear()
    monkeypatch.setattr("os.cpu_count", lambda: 64)

    # Act
    with ThreadPoolExecutor(2) as pool:
        pmap(abs, list(range(160)), workers=pool, options=QUIET)

    # Assert
    assert (sum(by_count), max(by_count)) == (160, 20)
    assert max(chunk_sizes) == 20


_Result = TypeVar("_Result")
_Params = ParamSpec("_Params")


class _CallbacklessFuture(Future[_Result]):
    def add_done_callback(self, fn: Callable[[Future[_Result]], object]) -> None:
        return


class _InlineExecutor(Executor):
    """Runs each task as it is submitted, so every chunk has finished."""

    def __init__(self, callbacks: bool = True) -> None:
        self.callbacks = callbacks

    def submit(
        self,
        fn: Callable[_Params, _Result],
        /,
        *args: _Params.args,
        **kwargs: _Params.kwargs,
    ) -> Future[_Result]:
        future: Future[_Result] = Future() if self.callbacks else _CallbacklessFuture()
        try:
            future.set_result(fn(*args, **kwargs))
        except ValueError as error:
            future.set_exception(error)
        return future


def test_pmap_counts_chunks_whose_callbacks_have_not_run_yet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    stream = io.StringIO()
    monkeypatch.setenv("PSNEK_MODE", "log")

    # Act
    pmap(
        abs,
        range(40),
        workers=_InlineExecutor(callbacks=False),
        options=Options(stream=stream),
    )

    # Assert
    assert "40/40" in stream.getvalue()
    assert "done" in stream.getvalue()


@pytest.mark.integration
def test_pmap_preserves_order_with_processes() -> None:
    # Arrange
    with ProcessPoolExecutor(2) as pool:
        # Act
        results = pmap(abs, [-3, 1, -2, 5], workers=pool, options=QUIET)

        # Assert: the caller's executor is left running
        assert pool.submit(abs, -1).result() == 1
    assert results == [3, 1, 2, 5]


def test_pmap_fails_fast_without_waiting_for_slow_chunks() -> None:
    # Arrange
    release = threading.Event()

    def work(item: int) -> int:
        if item == 0:
            release.wait(10)
            return item
        raise ValueError("fast failure")

    stream = io.StringIO()
    options = Options(stream=stream, leave=True)
    start = time.monotonic()

    # Act
    try:
        with pytest.raises(ValueError, match="fast failure"):
            pmap(work, [0, 1], workers=2, options=options)
        waited = time.monotonic() - start
    finally:
        release.set()

    # Assert
    assert waited < 5
    assert "failed" in stream.getvalue()


class _FalsyError(ValueError):
    def __len__(self) -> int:
        return 0


def test_pmap_raises_the_first_failure_in_input_order() -> None:
    # Arrange
    def work(item: int) -> int:
        if item in (2, 4):
            raise ValueError(f"item {item}")
        return item

    # Act / Assert
    with pytest.raises(ValueError, match="item 2"):
        pmap(work, [1, 2, 3, 4], workers=_InlineExecutor(), options=QUIET)
    with pytest.raises(ValueError, match="bad item"):
        pmap(_fail_at, range(10), workers=2, options=QUIET)


def test_pmap_stops_submitting_once_a_finished_chunk_failed() -> None:
    for error in (ValueError, _FalsyError):
        # Arrange
        called: list[int] = []

        def work(
            item: int, error: type[ValueError] = error, called: list[int] = called
        ) -> int:
            called.append(item)
            if item == 0:
                raise error("first item")
            return item

        # Act
        with pytest.raises(error, match="first item"):
            pmap(work, range(100), workers=_InlineExecutor(), options=QUIET)

        # Assert
        assert called == [0]


def test_log_handler_falls_back_to_sanitized_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    logger = logging.getLogger("psnek.test.fallback")
    handler = LogHandler()
    logger.addHandler(handler)

    # Act
    try:
        logger.warning("no bar active\x1b[2J")
    finally:
        logger.removeHandler(handler)

    # Assert
    assert capsys.readouterr().err == "warn  no bar active\n"


def test_log_tags_levels_and_times_while_write_prints_bare_text(
    log_stream: io.StringIO, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(_bar, "log_stamp", lambda when: "09:30:00")
    timed = Options(stream=log_stream, log_time=True)

    # Act
    with Bar(2, "train", options=Options(stream=log_stream)) as bar:
        bar.log("epoch", 1, "started")
        bar.write("saved checkpoint")
        bar.log("loss: spiked\n\nto 4.2", level=logging.WARNING)
        bar.log(level=_Level.NOTICE)
    with Bar(2, "eval", options=timed) as bar:
        bar.log("shard 7 failed", level=logging.ERROR)
        for named in ("warning", True):
            with pytest.raises(TypeError, match="level must be an int"):
                bar.log("named level", level=cast("int", named))

    # Assert: an empty message or line leaves no trailing spaces
    lines = [
        line
        for line in log_stream.getvalue().splitlines()
        if not line.startswith(("train:", "eval:"))
    ]
    assert lines == [
        "info  epoch 1 started",
        "saved checkpoint",
        "warn  loss: spiked",
        "",
        "      to 4.2",
        "info",
        "09:30:00 error shard 7 failed",
    ]


@pytest.fixture
def india_time(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Set the local time zone to one that is never UTC's, UTC+05:30."""
    if not hasattr(time, "tzset"):
        pytest.skip("the local time zone cannot be set on this platform")
    with monkeypatch.context() as zone:
        zone.setenv("TZ", "IST-05:30")
        time.tzset()
        yield
    time.tzset()


@pytest.mark.usefixtures("india_time")
def test_log_handler_tags_records_with_their_level_and_creation_time(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange: 2023-11-14 22:13:20 UTC
    record = logging.makeLogRecord(
        {"msg": "disk %s", "args": ("full",), "levelno": logging.CRITICAL}
    )
    record.created = 1_700_000_000.0
    stamp = "03:43:20"

    # Act
    LogHandler(log_time=True).emit(record)
    LogHandler().emit(record)

    # Assert
    assert capsys.readouterr().err.splitlines() == [
        f"{stamp} fatal disk full",
        "fatal disk full",
    ]


def test_log_handler_reports_its_own_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Arrange
    logger = logging.getLogger("psnek.test.handler_error")
    logger.propagate = False
    handler = LogHandler()
    logger.addHandler(handler)

    # Act
    try:
        logger.warning("%(missing)s", {"other": 1})
    finally:
        logger.removeHandler(handler)
        logger.propagate = True

    # Assert
    assert "Logging error" in capsys.readouterr().err


def test_log_handler_flushes_stdout_before_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    events: list[str] = []

    class Stdout(io.StringIO):
        def flush(self) -> None:
            events.append("flush stdout")

    class Stderr(io.StringIO):
        def write(self, text: str) -> int:
            events.append("write stderr")
            return len(text)

    monkeypatch.setenv("PSNEK_MODE", "log")
    monkeypatch.setattr("sys.stdout", Stdout())
    monkeypatch.setattr("sys.stderr", Stderr())
    record = logging.makeLogRecord({"msg": "saving", "levelno": logging.INFO})

    # Act
    LogHandler().emit(record)

    # Assert
    assert events[:2] == ["flush stdout", "write stderr"]


class _Ambiguous:
    def __eq__(self, other: object) -> bool:
        raise ValueError("truth value is ambiguous")

    __hash__ = None  # type: ignore[assignment]

    def __str__(self) -> str:
        return "array([1, 2])"


def test_panel_text_is_converted_with_str(terminal: FakeTerminal) -> None:
    # Arrange
    with Bar(total=1, options=Options(stream=terminal)) as bar:
        # Act
        bar.panel(42)
        number = _engine.ENGINES[(id(terminal), False)].panel_view()
        bar.panel(_Ambiguous())
        ambiguous = _engine.ENGINES[(id(terminal), False)].panel_view()
        bar.update()

    # Assert
    assert number == ["42"]
    assert ambiguous == ["array([1, 2])"]


def test_a_finished_bar_keeps_its_final_count() -> None:
    # Arrange
    bar = track(range(10), options=QUIET)
    iterator = iter(bar)
    next(iterator)
    next(iterator)

    # Act
    bar.interrupt()
    next(iterator)

    # Assert
    assert bar.outcome == "interrupted"
    assert bar.n == 1


def _interrupt_the_first_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = Bar._sample
    calls: list[int] = []

    def interrupt_once(self: Bar, now: float) -> object:
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return sample(self, now)

    monkeypatch.setattr(Bar, "_sample", interrupt_once)


def test_a_close_interrupted_while_sampling_can_be_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    retried = Bar(total=1, options=QUIET)
    updated = Bar(total=1, options=QUIET)
    _interrupt_the_first_sample(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        retried.close()

    # Act
    retried.close()
    updated.update()
    updated.close()

    # Assert
    assert retried.outcome == "interrupted"
    assert retried._engine is not None
    assert retried not in retried._engine.sources
    assert (updated.n, updated.outcome) == (1, "done")


def test_a_finished_loop_is_not_closed_again_when_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    flushed: list[TextIO] = []
    terminal = FakeTerminal()
    bar = track(iter(range(3)), options=Options(stream=terminal))
    iterator = iter(bar)
    next(iterator)
    monkeypatch.setattr(_engine, "flush_stdout", flushed.append)

    # Act
    for _ in iterator:
        pass
    del iterator

    # Assert
    assert bar.outcome == "done"
    assert flushed == [terminal]
