"""Bars wrapping an iterable (`track`, `zip`) or async one (`atrack`)."""

import builtins
import logging
import sys
import weakref
from collections.abc import (
    AsyncIterable,
    AsyncIterator,
    Callable,
    Iterable,
    Iterator,
    Sized,
)
from functools import partial
from itertools import chain, compress
from operator import call, index, length_hint
from typing import (
    Any,
    Generic,
    NoReturn,
    Self,
    SupportsIndex,
    TypedDict,
    TypeVar,
    Unpack,
    cast,
    overload,
)

from psnek._bar import Bar, Options, ending
from psnek._deferred import defer, describe
from psnek._env import once

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
_IMMUTABLE_ITERATORS = frozenset(
    type(iter(sample))
    for sample in ((), range(0), range(2**64, 2**64 + 1), "", "é", b"")
)


class TrackArgs(TypedDict, total=False):
    """Keyword arguments that `track` and `zip` pass on to their `Track`."""

    total: float | None
    options: Options | None


def _remaining(hint: Callable[[], int]) -> int | None:
    try:
        return hint()
    except OverflowError as error:
        once(logging.DEBUG, f"length hint too large: {describe(error)}")
        return None


def _total_of(iterable: object) -> float | None:
    if isinstance(iterable, Bar):
        return iterable.total
    if not isinstance(iterable, Sized):
        return None
    try:
        return len(iterable)
    except Exception as error:  # any len() failure means no total
        defer(logging.DEBUG, f"total unknown: len() failed: {describe(error)}")
        return None


class Track(Bar, Generic[T]):
    """A bar that wraps an iterable, shown once iteration begins; created by `track`."""

    __slots__ = ("_iterable",)
    _wraps_iterable = True

    def __init__(
        self,
        iterable: Iterable[T],
        desc: object = "",
        *,
        total: float | None = None,
        options: Options | None = None,
    ) -> None:
        """Wrap ``iterable``; the arguments and errors are as for `track`."""
        growable = total is None and type(iterable) is list
        if total is None:
            total = _total_of(iterable)
        self._iterable = iterable
        super().__init__(total, desc, options=options)
        if growable:
            self._sized = cast("Sized", iterable)

    def __len__(self) -> int:
        """Return the total as an int; TypeError if unknown or above ``sys.maxsize``."""
        total = self.total
        if total is None or total > sys.maxsize:
            raise TypeError("this bar has no total that fits a length")
        return int(total)

    def __iter__(self) -> Iterator[T]:
        """Iterate the wrapped items.

        Raises:
            RuntimeError: If the bar was already iterated.
        """
        self._target(claim=True)
        try:
            iterator = iter(self._iterable)
            self._iterable = ()
        except BaseException as error:
            self._finish(*ending(type(error)))
            raise
        remaining = partial(length_hint, iterator)
        immutable = type(iterator) in _IMMUTABLE_ITERATORS
        start = _remaining(remaining) if immutable else None
        if start is not None:
            self._fetched = lambda: start - remaining()
            source = iterator
        else:
            source = compress(iterator, self._count_pulls())
        tail = self._tail()
        weakref.finalize(tail, self._close_dropped).atexit = False
        return chain(source, tail)

    def _tail(self) -> Iterator[T]:
        self._settled = self._exhausted = True
        self._end_loop(None)
        yield from ()


class AsyncTrack(Bar, Generic[T]):
    """A bar that wraps an async iterable; created by `atrack`."""

    __slots__ = ("_iterable",)
    _wraps_iterable = True

    def __init__(
        self, iterable: AsyncIterable[T], desc: object = "", **args: Unpack[TrackArgs]
    ) -> None:
        """Wrap ``iterable``; the arguments and errors are as for `atrack`."""
        if not hasattr(iterable, "__aiter__"):
            name = type(iterable).__name__
            raise TypeError(f"atrack() needs an async iterable, got {name}")
        self._iterable = iterable
        super().__init__(desc=desc, **args)

    def __aiter__(self) -> AsyncIterator[T]:
        """Iterate the wrapped async iterable.

        Raises:
            RuntimeError: If the bar was already iterated.
            TypeError: If the iterable's ``__aiter__`` returns no async iterator.
        """
        self._target(claim=True)
        try:
            source = aiter(self._iterable)
        except BaseException as error:
            self._finish(*ending(type(error)))
            raise
        del self._iterable
        self._settled = True
        cursor = _AsyncCursor(self, source, self._count_pulls().__next__)
        weakref.finalize(cursor, self._close_dropped).atexit = False
        return cursor

    def __enter__(self) -> NoReturn:
        """Raise TypeError: async bars need ``async with``."""
        raise TypeError("use 'async with' for an async bar")

    async def __aenter__(self) -> Self:
        """Return this bar for an ``async with`` block, which closes it."""
        return Bar.__enter__(self)

    async def __aexit__(
        self, exc_type: type[BaseException] | None, *exc_info: object
    ) -> None:
        """Close the bar: failed on exceptions, interrupted on cancellation."""
        self._finish(*ending(exc_type))


class _AsyncCursor(Generic[T]):
    """The async iterator of an `AsyncTrack`; an item counts once the next is asked."""

    __slots__ = ("__weakref__", "_advance", "_bar", "_source", "_yielded")

    def __init__(
        self,
        bar: AsyncTrack[T],
        source: AsyncIterator[T],
        advance: Callable[[], object],
    ) -> None:
        """Iterate ``source`` for ``bar``; ``advance`` counts an item."""
        self._bar, self._source, self._advance = bar, source, advance
        self._yielded = False

    def __aiter__(self) -> Self:
        """Return this cursor."""
        return self

    async def __anext__(self) -> T:
        """Count the previous item and fetch the next."""
        bar = self._bar
        if self._yielded:
            self._yielded = False
            self._advance()
        try:
            item = await anext(self._source)
        except StopAsyncIteration:
            bar._exhausted = not bar._failed
            bar._end_loop(None)
            raise
        except BaseException as error:
            bar._failed = True
            bar._end_loop(*ending(type(error)))
            raise
        self._yielded = True
        return item


@overload
def track(
    iterable: SupportsIndex, desc: object = "", **args: Unpack[TrackArgs]
) -> Track[int]: ...
@overload
def track(
    iterable: Iterable[T], desc: object = "", **args: Unpack[TrackArgs]
) -> Track[T]: ...
def track(
    iterable: Iterable[T] | SupportsIndex, desc: object = "", **args: Unpack[TrackArgs]
) -> Track[T] | Track[int]:
    """Show progress while iterating.

    Args:
        iterable: Items to iterate, or an integer ``n`` meaning ``range(n)``.
        desc: Short description shown before the bar.
        **args: ``total`` (inferred when possible) and ``options``.

    Returns:
        An iterable bar yielding the same items.

    Raises:
        TypeError: If ``iterable``, ``total``, or ``options`` has the wrong type.
        ValueError: If ``total`` is negative or not finite.

    Example:
        >>> quiet = Options(disable=True)
        >>> for path in track(["a.txt", "b.txt"], "copy", options=quiet):
        ...     pass
    """
    if isinstance(iterable, bool):
        raise TypeError("track() needs an iterable or int, got bool")
    if not hasattr(iterable, "__iter__"):
        try:
            count = index(iterable)
        except TypeError:
            if not hasattr(iterable, "__getitem__"):
                name = type(iterable).__name__
                message = f"track() needs an iterable or int, got {name}"
                raise TypeError(message) from None
        else:
            return Track(range(count), desc, **args)
    return Track(cast("Iterable[T]", iterable), desc, **args)


def atrack(
    iterable: AsyncIterable[T], desc: object = "", **args: Unpack[TrackArgs]
) -> AsyncTrack[T]:
    """Show progress while iterating an async iterable with ``async for``.

    Args:
        iterable: Async items to iterate.
        desc: Short description shown before the bar.
        **args: ``total`` (if known) and ``options``, as for `track`.

    Returns:
        An async-iterable bar yielding the same items.

    Raises:
        TypeError: If ``iterable``, ``total``, or ``options`` has the wrong type.
        ValueError: As for `track`.
    """
    return AsyncTrack(iterable, desc, **args)


@overload
def zip(
    first: Iterable[T],
    second: Iterable[U],
    /,
    *,
    desc: object = "",
    strict: bool = False,
    **args: Unpack[TrackArgs],
) -> Track[tuple[T, U]]: ...
@overload
def zip(
    first: Iterable[T],
    second: Iterable[U],
    third: Iterable[V],
    /,
    *,
    desc: object = "",
    strict: bool = False,
    **args: Unpack[TrackArgs],
) -> Track[tuple[T, U, V]]: ...
@overload
def zip(
    *iterables: Iterable[Any],
    desc: object = "",
    strict: bool = False,
    **args: Unpack[TrackArgs],
) -> Track[tuple[Any, ...]]: ...
def zip(
    *iterables: Iterable[Any],
    desc: object = "",
    strict: bool = False,
    **args: Unpack[TrackArgs],
) -> Track[tuple[Any, ...]]:
    """Show progress while iterating in parallel, totaling the shortest sized iterable.

    Args:
        *iterables: Iterables to combine item by item.
        desc: Short description shown before the bar.
        strict: Raise ValueError when the iterables differ in length.
        **args: ``total`` and ``options``, as for `track`.

    Returns:
        An iterable bar yielding tuples with an item from each iterable.

    Raises:
        TypeError: As for `track`, or if an argument is not iterable.
        ValueError: As for `track`.

    Example:
        >>> quiet = Options(disable=True)
        >>> bar = zip("ab", [1, 2, 3], desc="pairs", options=quiet)
        >>> bar.total, list(bar)
        (2, [('a', 1), ('b', 2)])
    """
    for source in iterables:
        if not hasattr(source, "__iter__") and not hasattr(source, "__getitem__"):
            raise TypeError(f"expected iterables, got {type(source).__name__}")
    if args.get("total") is None:
        totals = [total for total in map(_total_of, iterables) if total is not None]
        args["total"] = min(totals, default=None)
    paired = partial(builtins.zip, *iterables, strict=strict)
    return Track(chain.from_iterable(map(call, (paired,))), desc, **args)
