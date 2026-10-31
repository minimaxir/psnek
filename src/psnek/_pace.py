"""A bar's pace: its rate, health, and remaining time, from its gaps between ticks."""

import math

from psnek._frame import Health

RATE_WINDOW, RATE_GAPS, WARMUP_GAPS, GAP_SMOOTHING = 5.0, 8, 3, 0.3
STALL_AFTER, STUCK_AFTER, STALL_SIGMAS, STUCK_SIGMAS = 5.0, 30.0, 3, 9
SHIFT_SIGMAS, SHIFT_GAPS = 3, 5
ETA_Z, ETA_TOLERANCE, ETA_RESOLUTION = 2.576, 0.1, 1.0

# Standard deviation of an exponentially weighted mean, per one of its samples.
_SMOOTHED_ERROR = math.sqrt(GAP_SMOOTHING / (2 - GAP_SMOOTHING))


class Pace:
    """Rate, health, and remaining-time statistics of one bar's progress.

    Attributes:
        changed_at: Monotonic time the count last changed.
        rate: Smoothed units per second, or None before it is known.

    Example:
        >>> pace = Pace(0, 0.0)
        >>> for second in range(1, 5):
        ...     pace.advance(second, float(second))
        >>> pace.rate, pace.eta(6, 4.0)
        (1.0, (6.0, False))
    """

    # fmt: off
    __slots__ = (
        "_count", "_fit_co", "_fit_place", "_fit_place_m2", "_fit_time",
        "_fit_time_m2", "_fit_units", "_gap_mean", "_gap_var", "_gaps", "_run",
        "_run_gaps", "_run_m2", "_run_mean", "_shift_start", "_shifted",
        "changed_at", "rate",
    )
    # fmt: on

    def __init__(self, count: float, now: float) -> None:
        """Start measuring from ``count`` units at monotonic time ``now``."""
        self.changed_at = now
        self.rate: float | None = None
        self._count = count
        self._gaps = 0
        self._gap_mean = self._gap_var = 0.0
        self._restart_run(count, now)

    def _restart_run(self, count: float, now: float) -> None:
        self._run = self._shift_start = (count, now)
        self._run_gaps = self._shifted = 0
        self._run_mean = self._run_m2 = 0.0
        self._fit_units = self._fit_place = self._fit_time = 0.0
        self._fit_co = self._fit_place_m2 = self._fit_time_m2 = 0.0

    def advance(self, count: float, now: float) -> None:
        """Fold in the ``count`` units observed at monotonic time ``now``."""
        units = count - self._count
        if not units:
            return
        gap = now - self.changed_at
        if units > 0:
            # Weighted Welford update of time per unit against unit position.
            per_unit = gap / units
            place = self._count - self._run[0] + units / 2
            fitted = self._fit_units
            if not fitted:
                self._fit_units, self._fit_place = units, place
                self._fit_time = per_unit
            else:
                share = units / (fitted + units)
                to_place = place - self._fit_place
                to_time = per_unit - self._fit_time
                place_mean = self._fit_place + to_place * share
                time_mean = self._fit_time + to_time * share
                weighted = units * to_place
                self._fit_co += weighted * (per_unit - time_mean)
                self._fit_place_m2 += weighted * (place - place_mean)
                self._fit_time_m2 += units * to_time * (per_unit - time_mean)
                self._fit_units = fitted + units
                self._fit_place, self._fit_time = place_mean, time_mean
        if self._gaps:
            self._observe(gap)
        else:
            self._gap_mean = self._run_mean = gap
            self._gaps = self._run_gaps = 1
        self._count, self.changed_at = count, now
        run_count, run_start = self._run
        window = max(RATE_WINDOW, RATE_GAPS * self._gap_mean)
        rate = self.rate
        if rate is None or now - run_start < window:
            rate = (count - run_count) / max(now - run_start, 1e-9)
        elif gap > 0:
            rate += (1 - math.exp(-gap / window)) * (units / gap - rate)
        self.rate = rate if math.isfinite(rate) else None

    def _observe(self, gap: float) -> None:
        """Add a later gap to both statistics; restart the run if the pace changed."""
        deviation = gap - self._gap_mean
        shift = GAP_SMOOTHING * deviation
        self._gap_mean += shift
        self._gap_var = (1 - GAP_SMOOTHING) * (self._gap_var + deviation * shift)
        self._gaps += 1
        self._run_gaps += 1
        delta = gap - self._run_mean
        self._run_mean += delta / self._run_gaps
        self._run_m2 += delta * (gap - self._run_mean)
        if self._run_gaps < WARMUP_GAPS:
            return
        # The recent mean must differ both significantly and by more than
        # the ETA tolerance, and keep doing so, to be a new pace.
        spread = math.sqrt(self._run_m2 / self._run_gaps)
        noise = SHIFT_SIGMAS * _SMOOTHED_ERROR * spread
        margin = max(noise, ETA_TOLERANCE * self._run_mean)
        if abs(self._gap_mean - self._run_mean) <= margin:
            self._shifted = 0
            return
        if not self._shifted:
            self._shift_start = (self._count, self.changed_at)
        self._shifted += 1
        if self._shifted >= SHIFT_GAPS:
            self._restart_run(*self._shift_start)

    def health(self, now: float) -> Health:
        """Flag the wait since the last progress if it is an outlier among gaps."""
        if self._gaps < WARMUP_GAPS:
            return "running"
        idle = now - self.changed_at
        mean = self._gap_mean
        # Exponential (memoryless) gaps deviate by their mean; assuming no less
        # spread keeps steady loops from stalling on small slowdowns.
        spread = max(math.sqrt(self._gap_var), mean)
        if idle > max(STUCK_AFTER, mean + STUCK_SIGMAS * spread):
            return "stuck"
        if idle > max(STALL_AFTER, mean + STALL_SIGMAS * spread):
            return "stalled"
        return "running"

    def eta(self, remaining: float, now: float) -> tuple[float | None, bool]:
        """Estimate the seconds ``remaining`` units take, as of `advance` at ``now``.

        Returns:
            The estimate (None while the rate is unknown) and whether it is approximate.
        """
        if remaining <= 0:
            return 0.0, False
        rate = self.rate
        if rate is None or rate <= 0:
            return None, False
        gap = self._gap_mean
        # The current step counts down to its typical gap; past it, the
        # overrun is expected to last as long again (Gott's median estimate).
        recent = max(remaining / rate, gap) - gap + abs(gap - (now - self.changed_at))
        start_count, start = self._run
        done = self._count - start_count
        gaps, mean = self._run_gaps, self._run_mean
        if gaps < WARMUP_GAPS or done <= 0 or mean <= 0:
            return recent, True
        variation = self._run_m2 / gaps / mean**2
        trust = 1 / (1 + variation)
        overall = remaining * (now - start) / done
        estimate = trust * recent + (1 - trust) * overall
        left = max(remaining / max(rate * gap, 1e-9), 1.0)
        variance = (
            estimate**2 * variation * (1 / left + 1 / gaps)
            + trust * (1 - trust) * (recent - overall) ** 2
        )
        spread = math.sqrt(variance)
        place_m2 = self._fit_place_m2
        if place_m2 > 0:
            # Squared slope plus its variance: the error of assuming no drift.
            slope = self._fit_co / place_m2
            residual = max(self._fit_time_m2 - self._fit_co * slope, 0.0)
            drift = slope * slope + residual / (gaps - 2) / place_m2
            spread = max(spread, remaining * remaining / 2 * math.sqrt(drift))
        tolerance = max(ETA_TOLERANCE * estimate, ETA_RESOLUTION)
        return estimate, self._shifted > 0 or ETA_Z * spread > tolerance
