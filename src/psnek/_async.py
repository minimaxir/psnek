"""Bars for concurrent asyncio work: `amap` and `gather` (asyncio imported on use)."""

from collections.abc import Awaitable, Callable, Iterable, Sized
from typing import Literal, TypeVar, cast, overload

from psnek._bar import Bar, Options

T = TypeVar("T")
R = TypeVar("R")


async def amap(
    function: Callable[[T], Awaitable[R]],
    iterable: Iterable[T],
    desc: object = "",
    *,
    workers: int | None = None,
    options: Options | None = None,
) -> list[R]:
    """Await ``function`` on every item, at most ``workers`` at once, preserving order.

    Args:
        function: Async callable applied to each item.
        iterable: Input items; one without ``len()`` is read into a list first.
        desc: Short description shown before the bar.
        workers: Maximum number of concurrent calls (default: no limit).
        options: Presentation settings for the bar.

    Returns:
        Results in input order.

    Raises:
        TypeError: If ``workers`` is not None or an int, or ``options`` not `Options`.
        ValueError: If ``workers`` is less than 1.
        Exception: The first failure of ``function``, once running calls are cancelled.

    Example:
        >>> import asyncio
        >>> async def double(value: int) -> int:
        ...     return value * 2
        >>> quiet = Options(disable=True)
        >>> asyncio.run(amap(double, [1, 2, 3], workers=2, options=quiet))
        [2, 4, 6]
    """
    import asyncio  # noqa: PLC0415

    if not isinstance(workers, int | None) or isinstance(workers, bool):
        raise TypeError(f"workers must be an int, not {workers!r}")
    if workers is not None and workers < 1:
        raise ValueError("workers must be at least 1")
    items = iterable if isinstance(iterable, Sized) else list(iterable)
    total = len(items)
    results: list[R | None] = [None] * total
    pending = zip(range(total), items, strict=False)
    with Bar(total, desc, options=options) as bar:

        async def work() -> None:
            for position, item in pending:
                results[position] = await function(item)
                bar.update()

        try:
            async with asyncio.TaskGroup() as group:
                for _ in range(min(workers or total, total)):
                    group.create_task(work())
        except BaseExceptionGroup as failures:
            failure = failures.exceptions[0]
        else:
            return cast("list[R]", results)
        raise failure


@overload
async def gather(
    *awaitables: Awaitable[R],
    desc: object = "",
    return_exceptions: Literal[False] = False,
    options: Options | None = None,
) -> list[R]: ...
@overload
async def gather(
    *awaitables: Awaitable[R],
    desc: object = "",
    return_exceptions: bool,
    options: Options | None = None,
) -> list[R | BaseException]: ...
async def gather(
    *awaitables: Awaitable[R],
    desc: object = "",
    return_exceptions: bool = False,
    options: Options | None = None,
) -> list[R] | list[R | BaseException]:
    """Run awaitables like ``asyncio.gather``, cancelling the rest after a failure.

    Args:
        *awaitables: Coroutines, tasks, or futures to run.
        desc: Short description shown before the bar.
        return_exceptions: Return exceptions in place of results instead of raising.
        options: Presentation settings for the bar.

    Returns:
        Results in argument order.

    Raises:
        TypeError: If ``options`` is not an `Options`, or an argument is not awaitable.
        Exception: Without ``return_exceptions``, the first exception raised.

    Example:
        >>> import asyncio
        >>> jobs = asyncio.sleep(0, "a"), asyncio.sleep(0, "b")
        >>> asyncio.run(gather(*jobs, options=Options(disable=True)))
        ['a', 'b']
    """
    import asyncio  # noqa: PLC0415

    with Bar(len(awaitables), desc, options=options) as bar:

        async def counted(awaitable: Awaitable[R]) -> R:
            try:
                result = await awaitable
            except Exception:
                bar.update()
                raise
            bar.update()
            return result

        futures = [asyncio.ensure_future(counted(each)) for each in awaitables]
        try:
            return await asyncio.gather(*futures, return_exceptions=return_exceptions)
        except BaseException:
            unfinished = [future for future in futures if future.cancel()]
            if unfinished:
                await asyncio.wait(unfinished)
            raise
