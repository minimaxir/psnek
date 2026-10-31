"""Tests for `amap` and `gather`, the bars over concurrent asyncio work."""

import asyncio
import io
from collections.abc import Awaitable, Iterator
from typing import cast

import pytest

from psnek import Bar, Options, amap, gather

QUIET = Options(disable=True)


@pytest.fixture
def log() -> io.StringIO:
    """A stream that receives plain log-mode status lines."""
    return io.StringIO()


@pytest.fixture
def logged(monkeypatch: pytest.MonkeyPatch, log: io.StringIO) -> Options:
    """Options that print each bar's log lines, including its final one, to `log`."""
    monkeypatch.setenv("PSNEK_MODE", "log")
    return Options(stream=log, leave=True)


def _final_line(log: io.StringIO) -> str:
    return log.getvalue().splitlines()[-1]


class _Gauge:
    """Counts calls that are running at once and records their peak."""

    def __init__(self) -> None:
        self.running = 0
        self.peak = 0

    async def double(self, value: int) -> int:
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0.001 * (value % 3))
        self.running -= 1
        return value * 2


def _items(count: int) -> Iterator[int]:
    yield from range(count)


def test_amap_preserves_order_and_counts_every_call(
    logged: Options, log: io.StringIO
) -> None:
    # Arrange
    gauge = _Gauge()

    growing = [1, 2]

    async def append_and_double(value: int) -> int:
        growing.append(value)
        return await gauge.double(value)

    # Act
    from_range = asyncio.run(amap(gauge.double, range(30), "x", options=logged))
    from_generator = asyncio.run(amap(gauge.double, _items(5), options=QUIET))
    from_growing = asyncio.run(amap(append_and_double, growing, options=QUIET))
    empty = asyncio.run(amap(gauge.double, [], options=QUIET))

    # Assert
    assert from_range == [value * 2 for value in range(30)]
    assert from_generator == [0, 2, 4, 6, 8]
    assert from_growing == [2, 4]
    assert empty == []
    assert "30/30" in _final_line(log)
    assert "done" in _final_line(log)


def test_amap_runs_at_most_workers_calls_in_as_many_tasks() -> None:
    # Arrange
    limited, unlimited = _Gauge(), _Gauge()
    task_counts: list[int] = []

    async def count_tasks(value: int) -> int:
        task_counts.append(len(asyncio.all_tasks()))
        return await limited.double(value)

    async def run() -> None:
        await amap(count_tasks, range(100), workers=4, options=QUIET)
        await amap(unlimited.double, range(100), options=QUIET)

    # Act
    asyncio.run(run())

    # Assert: the main task plus four workers
    assert limited.peak == 4
    assert max(task_counts) == 5
    assert unlimited.peak == 100


class _ChainedError(ValueError):
    pass


def test_amap_raises_the_first_failure_and_cancels_the_rest(
    logged: Options, log: io.StringIO
) -> None:
    # Arrange
    started: list[int] = []
    cancelled: list[int] = []

    async def work(value: int) -> int:
        started.append(value)
        if value == 1:
            await asyncio.sleep(0)
            try:
                raise KeyError(value)
            except KeyError as cause:
                raise _ChainedError("bad item") from cause
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(value)
            raise
        return value

    async def run() -> tuple[BaseException | None, list[int]]:
        with pytest.raises(_ChainedError, match="bad item") as raised:
            await amap(work, range(100), workers=3, options=logged)
        return raised.value.__cause__, sorted(cancelled)

    # Act: read the cancellations before asyncio.run cancels leftover tasks
    cause, cancelled_in_loop = asyncio.run(run())

    # Assert
    assert isinstance(cause, KeyError)
    assert started == [0, 1, 2]
    assert cancelled_in_loop == [0, 2]
    assert "failed" in _final_line(log)
    assert " failed _ChainedError " in _final_line(log)


def test_amap_is_interrupted_when_its_task_is_cancelled(
    logged: Options, log: io.StringIO
) -> None:
    # Arrange
    cancelled: list[int] = []

    async def work(value: int) -> int:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(value)
            raise
        return value

    async def run() -> list[int]:
        task = asyncio.ensure_future(amap(work, range(5), options=logged))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return sorted(cancelled)

    # Act: read the cancellations before asyncio.run cancels leftover tasks
    cancelled_in_loop = asyncio.run(run())

    # Assert
    assert cancelled_in_loop == [0, 1, 2, 3, 4]
    assert "interrupted" in _final_line(log)
    assert " interrupted cancelled " in _final_line(log)


@pytest.mark.parametrize(
    ("workers", "error"),
    [(0, ValueError), (True, TypeError), (cast("int", "4"), TypeError)],
)
def test_amap_rejects_invalid_worker_counts(
    workers: int, error: type[Exception]
) -> None:
    # Arrange
    async def identity(value: int) -> int:
        return value

    # Act / Assert
    with pytest.raises(error, match="workers"):
        asyncio.run(amap(identity, [1], workers=workers, options=QUIET))


async def _after(delay: float, value: int) -> int:
    await asyncio.sleep(delay)
    return value


def test_gather_preserves_order_across_awaitable_kinds(
    logged: Options, log: io.StringIO
) -> None:
    # Arrange
    async def run() -> list[int]:
        future = asyncio.get_running_loop().create_future()
        future.set_result(3)
        task = asyncio.ensure_future(_after(0.001, 2))
        return await gather(
            _after(0.005, 1), task, future, desc="mixed", options=logged
        )

    # Act
    results = asyncio.run(run())
    empty: list[int] = asyncio.run(gather(options=QUIET))

    # Assert
    assert results == [1, 2, 3]
    assert empty == []
    assert "3/3" in _final_line(log)
    assert "done" in _final_line(log)


def test_gather_raises_the_first_failure_and_cancels_the_rest(
    logged: Options, log: io.StringIO
) -> None:
    # Arrange
    cancelled: list[int] = []

    async def slow(value: int) -> int:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(value)
            raise
        return value

    async def fail() -> int:
        await asyncio.sleep(0)
        raise ValueError("bad request")

    async def run() -> list[int]:
        with pytest.raises(ValueError, match="bad request"):
            await gather(slow(0), fail(), slow(2), options=logged)
        return sorted(cancelled)

    # Act: read the cancellations before asyncio.run cancels leftover tasks
    cancelled_in_loop = asyncio.run(run())

    # Assert: the cancelled calls are not counted as done
    assert cancelled_in_loop == [0, 2]
    assert "1/3" in _final_line(log)
    assert "failed" in _final_line(log)
    assert " failed ValueError " in _final_line(log)


def test_gather_can_return_exceptions_in_place(
    logged: Options, log: io.StringIO
) -> None:
    # Arrange
    async def fail() -> int:
        raise ValueError("bad request")

    # Act
    results = asyncio.run(
        gather(_after(0, 1), fail(), return_exceptions=True, options=logged)
    )

    # Assert
    assert results[0] == 1
    assert isinstance(results[1], ValueError)
    assert "2/2" in _final_line(log)
    assert "done" in _final_line(log)


def test_gather_is_interrupted_when_its_task_is_cancelled(
    logged: Options, log: io.StringIO
) -> None:
    # Arrange
    async def run() -> list[bool]:
        children = [asyncio.ensure_future(_after(10, value)) for value in range(3)]
        task = asyncio.ensure_future(gather(*children, options=logged))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return [child.cancelled() for child in children]

    # Act: read the children's state before asyncio.run cancels leftover tasks
    cancelled_in_loop = asyncio.run(run())

    # Assert
    assert cancelled_in_loop == [True, True, True]
    assert "interrupted" in _final_line(log)
    assert " interrupted cancelled " in _final_line(log)


def test_gather_rejects_non_awaitables_and_cancels_what_it_started() -> None:
    # Arrange
    async def run() -> bool:
        task = asyncio.ensure_future(_after(10, 1))
        not_awaitable = cast("Awaitable[int]", 5)
        with pytest.raises(TypeError):
            await gather(task, not_awaitable, options=QUIET)
        return task.cancelled()

    # Act: read the state before asyncio.run cancels leftover tasks
    cancelled_in_loop = asyncio.run(run())

    # Assert
    assert cancelled_in_loop


def test_bars_started_inside_concurrent_calls_finish_cleanly(
    logged: Options, log: io.StringIO
) -> None:
    # Arrange
    async def download(value: int) -> int:
        with Bar(1, f"file {value}", options=logged) as bar:
            await asyncio.sleep(0.001)
            bar.update()
        return value

    # Act
    results = asyncio.run(amap(download, range(4), "files", options=logged))

    # Assert
    assert results == [0, 1, 2, 3]
    assert log.getvalue().count("done") == 5
