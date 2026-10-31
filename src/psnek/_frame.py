"""Bar states and the immutable `Frame` snapshot that every renderer draws."""

import math
import sys
from dataclasses import dataclass
from typing import Literal, get_args

from psnek._style import is_rgb
from psnek._text import SCALES, Scale, is_text_pair

Outcome = Literal["done", "failed", "interrupted"]
Health = Literal["running", "stalled", "stuck"]
State = Health | Outcome
STATES: frozenset[str] = frozenset((*get_args(Health), *get_args(Outcome)))
UNHEALTHY: frozenset[str] = frozenset(("stalled", "stuck"))
ENDED: tuple[Outcome, ...] = get_args(Outcome)
STOPPED = ("failed", "interrupted")
_LARGEST = sys.float_info.max


@dataclass(frozen=True, slots=True)
class Frame:
    """An immutable snapshot of everything needed to draw one bar.

    Attributes:
        n: Completed units.
        total: Expected units, or None when unknown.
        elapsed: Seconds since the bar started.
        rate: Smoothed units per second, or None before it is known.
        desc: Description.
        unit: Unit label.
        scale: Number prefix scaling.
        status: ``(key, value)`` pairs shown after the metrics.
        state: Lifecycle / health state.
        idle: Seconds since progress last changed.
        phase: Animation clock in seconds.
        branches: Per nesting level, whether that level's bar has siblings below.
        estimate: Seconds remaining from the bar's pace, or None to use ``rate``.
        approximate: Whether the ETA is uncertain, drawn after a ``~``.
        height: Rows the bar spans.
        children: Whether nested bars are drawn below, linked by a guide.
        cause: Why a failed or interrupted bar ended, such as ``Ctrl-C``.
        gradient: Color running and finished bars with the palette gradient, with
            these RGB stops instead, or (False) in the default foreground color.

    Raises:
        ValueError: If a number is not finite or in range, or a name is unknown.
        TypeError: If a field has the wrong type.
    """

    n: float = 0
    total: float | None = None
    elapsed: float = 0.0
    rate: float | None = None
    desc: str = ""
    unit: str = "it"
    scale: Scale = "none"
    status: tuple[tuple[str, str], ...] = ()
    state: State = "running"
    idle: float = 0.0
    phase: float = 0.0
    branches: tuple[bool, ...] = ()
    estimate: float | None = None
    approximate: bool = False
    height: int = 1
    children: bool = False
    cause: str = ""
    gradient: bool | tuple[int, ...] = True

    def __post_init__(self) -> None:
        """Reject values that cannot be drawn."""
        texts = (self.desc, self.unit, self.cause)
        status, branches = self.status, self.branches
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("frame desc, unit, and cause must be strings")
        if not isinstance(status, tuple) or not all(map(is_text_pair, status)):
            raise TypeError("frame status must be a tuple of (str, str) pairs")
        if not isinstance(branches, tuple) or {*map(type, branches)} - {bool}:
            raise TypeError("frame branches must be a tuple of bools")
        if type(self.approximate) is not bool or type(self.children) is not bool:
            raise TypeError("frame approximate and children must be bools")
        gradient = self.gradient
        stops = isinstance(gradient, tuple) and gradient and all(map(is_rgb, gradient))
        if type(gradient) is not bool and not stops:
            raise TypeError("frame gradient must be a bool or a tuple of RGB integers")
        if type(self.height) is not int:
            raise TypeError("frame height must be an int")
        if self.height < 1:
            raise ValueError("frame height must be at least 1")
        if self.scale not in SCALES:
            raise ValueError(f"frame scale must be one of {SCALES}")
        if self.state not in STATES:
            raise ValueError(f"frame state must be one of {sorted(STATES)}")
        optional = [x for x in (self.total, self.rate, self.estimate) if x is not None]
        numbers = (self.n, self.elapsed, self.idle, self.phase, *optional)
        try:
            finite = all(map(math.isfinite, numbers))
        except (TypeError, OverflowError) as error:
            raise ValueError(f"frame numbers must be real numbers: {error}") from error
        if not finite:
            raise ValueError("frame numbers must be finite")
        if self.total is not None and self.total < 0:
            raise ValueError("frame total must be non-negative")

    @property
    def fraction(self) -> float | None:
        """Completed fraction clamped to ``[0, 1]``, or None if total is unknown."""
        if self.total is None:
            return None
        if self.total <= 0:
            return 1.0 if self.n > 0 or self.state == "done" else 0.0
        return min(max(self.n / self.total, 0.0), 1.0)

    @property
    def complete(self) -> bool:
        """Whether the count reached a known total."""
        return self.total is not None and self.n >= self.total

    @property
    def full(self) -> bool:
        """Whether 100% may be shown: the count reached the total or it is done."""
        return self.complete or self.state == "done"

    @property
    def eta(self) -> float | None:
        """Estimated seconds remaining (at most the largest float), or None."""
        if self.total is None:
            return None
        if self.estimate is not None:
            return max(self.estimate, 0.0)
        if self.rate is None or self.rate <= 0:
            return None
        return min(max(self.total - self.n, 0) / self.rate, _LARGEST)
