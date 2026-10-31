"""Overhead of psnek versus tqdm, printed as Markdown tables.

Every run is ``_worker.py`` in a fresh interpreter with a scrubbed
environment, one after another; the libraries take turns within each round,
and the rounds cycle through every order of the three (use a multiple of six
repeats for a balanced cycle), so drift does not favor any of them. For each
library the fastest run is reported, with the CPU time and output volume of
that same run. Imports happen before timing starts. Both libraries
write to the same ``/dev/null`` sink at their default refresh cadence (tqdm:
10 Hz, psnek: 12 Hz live mode, with true color forced so psnek's styling runs
the same way on every machine), so the numbers measure the per-item hot path
plus each library's background bookkeeping. Process CPU time is reported next
to wall time, so work moved to a render thread is not hidden, and so is the
progress output written per unit, since the libraries draw at different
moments: tqdm draws every bar when it is created, while psnek draws on render
ticks, so a nested bar that ends within one tick writes nothing, and its
kept final lines are colored cell by cell. The first top-level psnek bar
starts a render thread, which later bars reuse while it lingers. The number
of items each run processed is checked after the timed region (psnek's check
that each bar ended as done runs inside it, which only adds to psnek's time).
psnek runs with ``capture=False`` and ``PSNEK_MODE=live`` so that, like tqdm,
it neither redirects ``print()`` nor scans the environment for CI and agents.
Since ``/dev/null`` is not a terminal and ``COLUMNS``/``LINES`` are scrubbed,
psnek lays bars out at its 80x24 fallback size, querying it on each draw.

The parallel map table compares end-to-end time on one worker thread: the
stdlib ``Executor.map`` (one future per item, which is what tqdm's
``thread_map`` uses), tqdm's ``thread_map``, and ``psnek.pmap``, which sends
items in chunks.

The async table compares end-to-end time for calls that each yield to the
event loop once: ``asyncio.gather``, ``tqdm.asyncio.tqdm.gather``, and
``psnek.gather`` over every call at once; and, limited to 16 concurrent calls,
``asyncio.gather`` and tqdm's ``gather`` over semaphore-wrapped calls (the
usual idiom) against ``psnek.amap`` with 16 workers.

Usage::

    uv run --group bench python benchmarks/bench.py [--items N] [--repeats R]
"""

import argparse
import itertools
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

WORKER = Path(__file__).with_name("_worker.py")

LIBRARIES = ("baseline", "tqdm", "psnek")
ITEM_KINDS = {
    "list": "iterate a list",
    "tuple": "iterate a tuple",
    "range": "iterate a range",
    "generator": "iterate a generator",
    "update": "manual update()",
}
BAR_KINDS = {
    "short": "top-level bar of 10 items",
    "nested": "nested bar of 10 items",
}
SCRUBBED_PREFIXES = ("PSNEK_", "TQDM_")
SCRUBBED_NAMES = {
    "NO_COLOR",
    "FORCE_COLOR",
    "COLORTERM",
    "TERM_PROGRAM",
    "TERM_PROGRAM_VERSION",
    "WT_SESSION",
    "CONEMUANSI",
    "COLUMNS",
    "LINES",
    "TERM",
}
BAR_ITEMS_DIVISOR = 1000
MAP_ITEMS_DIVISOR = 10
ASYNC_ITEMS_DIVISOR = 50
ASYNC_KINDS = {
    "gather": "gather every call at once",
    "amap": "map, 16 calls at a time",
}
MIN_ITEMS = ASYNC_ITEMS_DIVISOR * 10


Run = tuple[float, float, float]


def measure(library: str, kind: str, items: int) -> Run:
    """Run one benchmark case once in a fresh interpreter.

    Args:
        library: ``baseline``, ``tqdm``, or ``psnek``.
        kind: A key of `ITEM_KINDS`, `BAR_KINDS`, or `ASYNC_KINDS`, or ``pmap``.
        items: Items in the run.

    Returns:
        Wall-clock and CPU nanoseconds, and characters written, per unit
        (item or bar).

    Raises:
        subprocess.CalledProcessError: If the worker fails.
        ValueError: If the worker prints something other than three numbers.
    """
    command = [sys.executable, str(WORKER), library, kind, str(items)]
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in SCRUBBED_NAMES and not name.startswith(SCRUBBED_PREFIXES)
    }
    environment.update(
        PYTHONHASHSEED="0", PSNEK_MODE="live", FORCE_COLOR="1", COLORTERM="truecolor"
    )
    result = subprocess.run(  # noqa: S603
        command, capture_output=True, text=True, check=True, env=environment
    )
    wall, cpu, written = map(float, result.stdout.split())
    return wall, cpu, written


def _ratio(tqdm_cost: float, psnek_cost: float, words: tuple[str, str]) -> str:
    """Compare two costs as a factor, worded by whether psnek is cheaper."""
    if tqdm_cost <= 0 or psnek_cost <= 0:
        return "n/a (below noise)"
    if psnek_cost <= tqdm_cost:
        return f"{tqdm_cost / psnek_cost:.1f}x {words[0]}"
    return f"{psnek_cost / tqdm_cost:.1f}x {words[1]}"


def _best(kind: str, items: int, repeats: int) -> dict[str, Run]:
    """Measure every library in turns and keep each one's fastest run.

    Args:
        kind: A key of `ITEM_KINDS`, `BAR_KINDS`, or `ASYNC_KINDS`, or ``pmap``.
        items: Items per run.
        repeats: Rounds of runs, each in the next order of the libraries.

    Returns:
        The fastest run of each library, keyed in `LIBRARIES` order.
    """
    runs: dict[str, list[Run]] = {library: [] for library in LIBRARIES}
    orders = itertools.cycle(itertools.permutations(LIBRARIES))
    for order in itertools.islice(orders, repeats):
        for library in order:
            runs[library].append(measure(library, kind, items))
    return {library: min(found) for library, found in runs.items()}


def _table(kinds: dict[str, str], items: int, repeats: int, unit: str) -> None:
    """Print one overhead table, measuring each case sequentially.

    Args:
        kinds: Cases to run, with their labels.
        items: Items per repeat.
        repeats: Repeats per case.
        unit: Unit label for the cells, such as ``ns`` or ``us``.
    """
    scale = 1e-3 if unit == "us" else 1.0
    print(
        "| case | bare | tqdm | psnek | psnek overhead vs tqdm overhead "
        "| tqdm CPU | psnek CPU | tqdm output | psnek output |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for kind, label in kinds.items():
        runs = _best(kind, items, repeats)
        (base, _, _), (tqdm, tqdm_cpu, tqdm_out), (psnek, psnek_cpu, psnek_out) = (
            runs.values()
        )
        ratio = _ratio(tqdm - base, psnek - base, ("less", "more"))
        cells = [base, tqdm, psnek, tqdm_cpu, psnek_cpu]
        times = [f"{value * scale:.1f} {unit}" for value in cells]
        print(
            f"| {label} | {' | '.join(times[:3])} | {ratio} | {' | '.join(times[3:])} "
            f"| {_chars(tqdm_out)} | {_chars(psnek_out)} |"
        )


def _chars(count: float) -> str:
    """Format characters written per unit compactly."""
    return f"{count:.3g} ch"


def _map_table(items: int, repeats: int) -> None:
    """Print the end-to-end parallel map comparison.

    Args:
        items: Items per repeat.
        repeats: Repeats per case.
    """
    runs = _best("pmap", items, repeats)
    (base, _, _), (tqdm, _, _), (psnek, _, _) = runs.values()
    print("| case | Executor.map | tqdm thread_map | psnek pmap | psnek vs tqdm |")
    print("|---|---:|---:|---:|---:|")
    print(
        f"| map over 1 thread | {base:.1f} ns | {tqdm:.1f} ns | {psnek:.1f} ns "
        f"| {_ratio(tqdm, psnek, ('faster', 'slower'))} |"
    )


def _async_table(items: int, repeats: int) -> None:
    """Print the end-to-end async comparison, measuring each case sequentially.

    Args:
        items: Items per repeat.
        repeats: Repeats per case.
    """
    print("| case | asyncio | tqdm gather | psnek | psnek vs tqdm |")
    print("|---|---:|---:|---:|---:|")
    for kind, label in ASYNC_KINDS.items():
        runs = _best(kind, items, repeats)
        (base, _, _), (tqdm, _, _), (psnek, _, _) = runs.values()
        print(
            f"| {label} | {base:.1f} ns | {tqdm:.1f} ns | {psnek:.1f} ns "
            f"| {_ratio(tqdm, psnek, ('faster', 'slower'))} |"
        )


def main() -> None:
    """Run every case sequentially and print Markdown tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=5_000_000)
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    if args.items < MIN_ITEMS or args.repeats < 1:
        parser.error(f"--items must be at least {MIN_ITEMS} and --repeats at least 1")
    build = "free-threaded" if sysconfig.get_config_var("Py_GIL_DISABLED") else "GIL"
    bar_items = max(args.items // BAR_ITEMS_DIVISOR, 100) * 10
    print(f"Python {sys.version.split()[0]} ({build}), {args.items:,} items\n")
    print("Per item:\n")
    _table(ITEM_KINDS, args.items, args.repeats, "ns")
    print(f"\nPer bar ({bar_items // 10:,} bars):\n")
    _table(BAR_KINDS, bar_items, args.repeats, "us")
    print("\nParallel map, end to end per item:\n")
    _map_table(args.items // MAP_ITEMS_DIVISOR, args.repeats)
    print("\nAsync, end to end per item:\n")
    _async_table(args.items // ASYNC_ITEMS_DIVISOR, args.repeats)


if __name__ == "__main__":
    main()
