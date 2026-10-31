"""Time one benchmark case in this interpreter and print its cost per unit.

Run by ``bench.py`` in a fresh process per run, so every run starts with
cold caches::

    python benchmarks/_worker.py LIBRARY KIND ITEMS

It prints the wall-clock and process CPU nanoseconds per unit, and the
characters of progress output written per unit: per item for loop, map, and
async kinds, per bar for the ``short`` and ``nested`` kinds.
"""

import asyncio
import operator
import os
import sys
import time
from collections.abc import Awaitable, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TextIO, cast

from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from tqdm.contrib.concurrent import thread_map

import psnek

Case = Callable[[], int]
BAR_ITEMS = 10
BAR_KINDS = ("short", "nested")
ASYNC_KINDS = ("gather", "amap")
LOOP_KINDS = ("list", "tuple", "range", "generator", "update", "pmap")
KINDS = (*LOOP_KINDS, *ASYNC_KINDS, *BAR_KINDS)
ASYNC_WORKERS = 16
LIBRARIES = ("baseline", "tqdm", "psnek")


class CountingSink:
    """A text stream that counts the characters written through it."""

    def __init__(self, stream: TextIO) -> None:
        """Wrap ``stream``.

        Args:
            stream: Where the text goes.
        """
        self.stream = stream
        self.written = 0

    def write(self, text: str) -> int:
        """Count and forward ``text``.

        Args:
            text: Text to write.

        Returns:
            The number of characters written.
        """
        self.written += len(text)
        return self.stream.write(text)

    def __getattr__(self, name: str) -> object:
        """Forward everything else to the wrapped stream.

        Args:
            name: Attribute name.

        Returns:
            The wrapped stream's attribute.
        """
        return getattr(self.stream, name)


class Baseline:
    """A no-op progress object, measuring the cost of a method call alone."""

    def update(self, n: int = 1) -> None:
        """Do nothing.

        Args:
            n: Ignored increment.
        """


async def _negate(value: int) -> int:
    """Yield to the event loop once, like a call waiting on I/O, then negate."""
    await asyncio.sleep(0)
    return -value


def _limited(data: list[int]) -> list[Awaitable[int]]:
    """Wrap each call in a semaphore, the usual way to limit ``asyncio.gather``.

    Args:
        data: Items to process.

    Returns:
        One coroutine per item, at most `ASYNC_WORKERS` of them running at once.
    """
    limit = asyncio.Semaphore(ASYNC_WORKERS)

    async def call(value: int) -> int:
        async with limit:
            return await _negate(value)

    return [call(value) for value in data]


async def _gather(awaitables: Iterable[Awaitable[int]]) -> list[int]:
    """Await ``awaitables`` with ``asyncio.gather``, from a coroutine."""
    return await asyncio.gather(*awaitables)


def _source(data: list[int], kind: str) -> Callable[[], Iterable[int]]:
    """Build the iterable for a loop kind once, outside the timed region.

    Args:
        data: Items to process.
        kind: Benchmark kind.

    Returns:
        A callable returning the iterable for one run; only a generator is
        created per run.
    """
    if kind == "generator":
        return lambda: (value for value in data)
    built = {"tuple": tuple(data), "range": range(len(data))}.get(kind, data)
    return lambda: built


def _baseline(kind: str, data: list[int]) -> Case:
    """Build the loop without any progress reporting.

    Args:
        kind: Benchmark kind.
        data: Items to process.

    Returns:
        A callable running the loop once and returning the items processed.
    """
    inner = data[:BAR_ITEMS]
    source = _source(data, kind)

    def run() -> int:
        if kind == "pmap":
            with ThreadPoolExecutor(1) as pool:
                return len(list(pool.map(operator.neg, data)))
        if kind == "gather":
            return len(asyncio.run(_gather(map(_negate, data))))
        if kind == "amap":
            return len(asyncio.run(_gather(_limited(data))))
        if kind in BAR_KINDS:
            bars = len(data) // BAR_ITEMS
            for _ in range(bars):
                for _ in inner:
                    pass
            return bars * BAR_ITEMS
        if kind == "update":
            counter = Baseline()
            for _ in data:
                counter.update()
            return len(data)
        for _ in source():
            pass
        return len(data)

    return run


def _tqdm(kind: str, data: list[int], sink: TextIO) -> Case:
    """Build the loop reporting progress with tqdm.

    Args:
        kind: Benchmark kind.
        data: Items to process.
        sink: Text stream receiving progress output.

    Returns:
        A callable running the loop once and returning the items processed.
    """
    inner = data[:BAR_ITEMS]
    source = _source(data, kind)

    def run() -> int:
        if kind == "pmap":
            results = thread_map(operator.neg, data, max_workers=1, file=sink)
            return len(results)
        if kind in ASYNC_KINDS:
            calls = map(_negate, data) if kind == "gather" else _limited(data)
            return len(asyncio.run(tqdm_asyncio.gather(*calls, file=sink)))
        if kind in BAR_KINDS:
            bars = range(len(data) // BAR_ITEMS)
            outer = tqdm(bars, file=sink) if kind == "nested" else bars
            processed = 0
            for _ in outer:
                progress = tqdm(inner, file=sink, leave=kind == "short")
                for _ in progress:
                    pass
                processed += progress.n
            return processed
        if kind == "update":
            with tqdm(total=len(data), file=sink) as counter:
                for _ in data:
                    counter.update()
            return int(counter.n)
        progress = tqdm(source(), total=len(data), file=sink)
        for _ in progress:
            pass
        return int(progress.n)

    return run


def _psnek(kind: str, data: list[int], sink: TextIO) -> Case:
    """Build the loop reporting progress with psnek.

    Args:
        kind: Benchmark kind.
        data: Items to process.
        sink: Text stream receiving progress output.

    Returns:
        A callable running the loop once and returning the items processed;
        it raises RuntimeError if a bar did not finish as done.
    """
    options = psnek.Options(stream=sink, capture=False)
    inner = data[:BAR_ITEMS]
    source = _source(data, kind)

    def finished(bar: psnek.Bar) -> int:
        if bar.outcome != "done":
            raise RuntimeError(f"psnek finished as {bar.outcome}")
        return int(bar.n)

    def run() -> int:
        if kind == "pmap":
            return len(psnek.pmap(operator.neg, data, workers=1, options=options))
        if kind == "gather":
            calls = map(_negate, data)
            return len(asyncio.run(psnek.gather(*calls, options=options)))
        if kind == "amap":
            mapped = psnek.amap(_negate, data, workers=ASYNC_WORKERS, options=options)
            return len(asyncio.run(mapped))
        if kind in BAR_KINDS:
            bars = range(len(data) // BAR_ITEMS)
            outer = psnek.track(bars, options=options) if kind == "nested" else bars
            processed = 0
            for _ in outer:
                progress = psnek.track(inner, options=options)
                for _ in progress:
                    pass
                processed += finished(progress)
            return processed
        if kind == "update":
            with psnek.Bar(total=len(data), options=options) as bar:
                for _ in data:
                    bar.update()
            return finished(bar)
        progress = psnek.track(source(), total=len(data), options=options)
        for _ in progress:
            pass
        return finished(progress)

    return run


def _case(library: str, kind: str, data: list[int], sink: TextIO) -> Case:
    """Build the timed loop for one library and kind.

    Args:
        library: ``baseline``, ``tqdm``, or ``psnek``.
        kind: ``list``, ``tuple``, ``range``, ``generator``, ``update``,
            ``short``, ``nested``, ``pmap``, ``gather``, or ``amap``.
        data: Items to process.
        sink: Text stream receiving progress output.

    Returns:
        A callable running the loop once and returning the items processed.
    """
    if library == "baseline":
        return _baseline(kind, data)
    return (_tqdm if library == "tqdm" else _psnek)(kind, data, sink)


def main() -> None:
    """Time the case named on the command line and print its cost per unit.

    Raises:
        SystemExit: If the arguments are not a library, a kind, and at least
            ten items.
        RuntimeError: If the run did not process every item.
    """
    try:
        library, kind, count = sys.argv[1:]
    except ValueError:
        raise SystemExit(__doc__) from None
    valid = library in LIBRARIES and kind in KINDS
    valid = valid and count.isascii() and count.isdigit()
    if not valid or int(count) < BAR_ITEMS:
        raise SystemExit(__doc__)
    items = int(count)
    data = list(range(items))
    units = items // BAR_ITEMS if kind in BAR_KINDS else items
    expected = units * BAR_ITEMS if kind in BAR_KINDS else items
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        sink = CountingSink(devnull)
        run = _case(library, kind, data, cast("TextIO", sink))
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        processed = run()
        cpu = time.process_time() - cpu_start
        wall = time.perf_counter() - wall_start
    if processed != expected:
        raise RuntimeError(f"{library} processed {processed} of {expected}")
    print(wall / units * 1e9, cpu / units * 1e9, sink.written / units)


if __name__ == "__main__":
    main()
