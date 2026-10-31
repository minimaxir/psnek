import bisect
import itertools
import math
import random
import statistics
from collections.abc import Callable, Iterator

import pytest

from psnek._pace import ETA_RESOLUTION, ETA_TOLERANCE, Pace

TICK = 1 / 12


def _ticks(durations: list[float]) -> Iterator[tuple[float, float, float, Pace]]:
    """Advance a pace at render ticks over units taking ``durations`` seconds.

    Yields:
        The time, the seconds truly left, the units left, and the pace.
    """
    finish = list(itertools.accumulate(durations))
    pace = Pace(0, 0.0)
    for tick in range(1, math.ceil(finish[-1] / TICK)):
        now = tick * TICK
        done = bisect.bisect_right(finish, now)
        pace.advance(done, now)
        yield now, finish[-1] - now, len(durations) - done, pace


def _uniform(rng: random.Random) -> list[float]:
    return [rng.uniform(2.4, 3.6) for _ in range(40)]


def _lognormal(rng: random.Random) -> list[float]:
    return [rng.lognormvariate(-1, 1) for _ in range(600)]


def test_rate_averages_then_smooths_and_holds_between_steps() -> None:
    # Arrange
    pace = Pace(0, 0.0)
    for second in range(1, 11):
        pace.advance(10 * second, float(second))
    averaged = pace.rate

    # Act
    pace.advance(130, 11.0)
    smoothed = pace.rate
    pace.advance(130, 20.0)

    # Assert
    assert averaged == pytest.approx(10)
    assert smoothed == pytest.approx(10 + (1 - math.exp(-1 / 8)) * 20)
    assert pace.rate == smoothed


def test_a_steady_loop_gets_an_exact_certain_eta_after_warm_up() -> None:
    # Arrange
    pace = Pace(0, 0.0)
    warming = []
    for step in range(1, 6):
        pace.advance(step, 10.0 * step)
        warming.append(pace.eta(12 - step, 10.0 * step)[1])

    # Act: four seconds into the sixth of twelve ten-second steps, then 15 past due
    estimate, approximate = pace.eta(7, 54.0)
    overdue, _ = pace.eta(7, 75.0)

    # Assert
    assert warming == [True, True, False, False, False]
    assert estimate == pytest.approx(66.0)
    assert not approximate
    assert overdue == pytest.approx(75.0)


def test_eta_edge_cases() -> None:
    # Arrange
    chunked = Pace(0, 0.0)
    for second in range(1, 6):
        chunked.advance(100 * second, float(second))

    # Act
    unknown = Pace(0, 0.0).eta(5, 1.0)
    finished = chunked.eta(0, 5.0)
    last_chunk, approximate = chunked.eta(10, 5.0)

    # Assert
    assert unknown == (None, False)
    assert finished == (0.0, False)
    assert last_chunk == pytest.approx(1.0)
    assert not approximate


def test_a_change_of_pace_is_marked_then_followed() -> None:
    # Arrange
    durations = [0.1] * 300 + [0.3] * 300

    # Act
    pending, followed = [], []
    for now, truth, units, pace in _ticks(durations):
        estimate, approximate = pace.eta(units, now)
        if estimate is None:
            continue
        if 30 < now < 32:
            pending.append(approximate)
        elif 36 < now < 40:
            followed.append(abs(estimate - truth) <= ETA_TOLERANCE * truth)

    # Assert
    assert any(pending)
    assert all(followed)


@pytest.mark.parametrize(
    ("make", "worst_median", "ever_plain"),
    [
        (lambda _: [0.02] * 3000, 0.01, True),
        (_uniform, 0.05, True),
        (lambda _: [10.0] * 12, 0.01, True),
        (lambda _: [15.0 if i % 50 == 49 else 0.1 for i in range(500)], 0.15, False),
        (_lognormal, 0.08, False),
        (lambda _: [0.05 + 0.2 * i / 1000 for i in range(1000)], 0.15, True),
        (lambda _: [0.15 - 0.1 * i / 1000 for i in range(1000)], 0.3, False),
        (lambda _: [0.1 * (1.5 + math.sin(i / 40)) for i in range(1000)], 0.3, True),
    ],
    ids=[
        "fast",
        "slow-noisy",
        "slow-steady",
        "bursty",
        "heavy-tailed",
        "slowing-down",
        "speeding-up",
        "oscillating",
    ],
)
def test_eta_is_accurate_and_unmarked_only_when_confident(
    make: Callable[[random.Random], list[float]],
    worst_median: float,
    ever_plain: bool,
) -> None:
    # Arrange
    durations = make(random.Random(0))
    length = sum(durations)

    # Act
    errors, marks, misses = [], [], 0
    for now, truth, units, pace in _ticks(durations):
        estimate, approximate = pace.eta(units, now)
        if estimate is None or not 0.1 * length < now < 0.9 * length:
            continue
        errors.append(abs(estimate - truth) / length)
        marks.append(approximate)
        tolerance = max(ETA_TOLERANCE * truth, ETA_RESOLUTION)
        misses += not approximate and abs(estimate - truth) > tolerance

    # Assert: a wrong ETA is shown without ~ at most 2% of the time
    assert statistics.median(errors) < worst_median
    assert (not all(marks)) is ever_plain
    assert misses <= 0.02 * len(errors)
