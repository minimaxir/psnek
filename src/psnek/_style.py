"""Styled text runs and their serialization to ANSI, HTML, and plain text."""

import html
import re
from collections.abc import Callable, Iterable
from dataclasses import astuple, dataclass
from enum import IntEnum
from functools import lru_cache
from typing import NamedTuple


class ColorDepth(IntEnum):
    """How many colors the output target can display (``NONE``: no escapes at all)."""

    NONE = 0
    EIGHT_BIT = 8
    TRUE_COLOR = 24


class Ink(NamedTuple):
    """Visual attributes of a text run: 24-bit color (None: default), bold, faint."""

    color: int | None = None
    bold: bool = False
    faint: bool = False


Run = tuple[str, Ink]
PanelLine = str | tuple[Run, ...]
PLAIN, BOLD, FAINT = Ink(), Ink(bold=True), Ink(faint=True)
_WHITE = 0xFFFFFF

_CUBE_LEVELS = (0, 95, 135, 175, 215, 255)
_GRAY_START, _GRAY_STEP, _GRAY_LEVELS = 8, 10, 24
_HEX = re.compile(r"#?([0-9A-Fa-f]{6})")


def _channels(color: int) -> tuple[int, int, int]:
    return (color >> 16) & 255, (color >> 8) & 255, color & 255


def is_rgb(color: object) -> bool:
    """Whether ``color`` is a 24-bit RGB integer."""
    return type(color) is int and 0 <= color <= _WHITE


def parse_hex(colors: tuple[str, ...]) -> tuple[int, ...]:
    """Parse ``"#RRGGBB"`` strings (the ``#`` is optional) into RGB integers.

    Args:
        colors: Hex color strings.

    Returns:
        The colors as 24-bit RGB integers, in order.

    Raises:
        ValueError: If ``colors`` is empty or a color is not six hex digits.

    Example:
        >>> [f"{color:06x}" for color in parse_hex(("#00948B", "669100"))]
        ['00948b', '669100']
    """
    matches = [_HEX.fullmatch(color) for color in colors]
    if not matches or None in matches:
        raise ValueError(f"colors must be one or more '#RRGGBB' strings: {colors}")
    return tuple(int(match[1], 16) for match in matches if match)


@dataclass(frozen=True, slots=True)
class Palette:
    """Colors for every bar state, keeping 3:1 contrast on common backgrounds.

    Attributes:
        gradient: Evenly spaced stops coloring the bar from left to right.
        stalled: Warning color while no progress is being made.
        failed: Error color for stuck and failed bars.
        interrupted: Neutral color for bars closed before completion.
        shimmer: Lighter highlight that glides along running bars.

    Raises:
        ValueError: If ``gradient`` is empty or a color is not a 24-bit RGB integer.
    """

    gradient: tuple[int, ...] = (0xA467E0, 0x4C7EBC, 0x1F8D9D, 0x1B976F)
    stalled: int = 0xBA7D0A
    failed: int = 0xE5534B
    interrupted: int = 0x878787
    shimmer: int = 0x508CEE

    def __post_init__(self) -> None:
        """Reject palettes that cannot be drawn."""
        if not isinstance(self.gradient, tuple) or not self.gradient:
            raise ValueError("palette gradient must be a non-empty tuple of colors")
        if not all(map(is_rgb, (*self.gradient, *astuple(self)[1:]))):
            raise ValueError("palette colors must be 24-bit RGB integers")


DEFAULT_PALETTE = Palette()


def blend(start: int, end: int, amount: float) -> int:
    """Interpolate from RGB color ``start`` to ``end`` by ``amount`` in ``[0, 1]``."""
    red, green, blue = (
        round(low * (1 - amount) + high * amount)
        for low, high in zip(_channels(start), _channels(end), strict=True)
    )
    return (red << 16) | (green << 8) | blue


def gradient_at(stops: tuple[int, ...], position: float) -> int:
    """Sample evenly spaced color ``stops`` at ``position`` in ``[0, 1]``."""
    if len(stops) == 1:
        return stops[0]
    scaled = min(max(position, 0.0), 1.0) * (len(stops) - 1)
    index = min(int(scaled), len(stops) - 2)
    return blend(stops[index], stops[index + 1], scaled - index)


def _distance(first: tuple[int, ...], second: tuple[int, ...]) -> int:
    return sum((one - two) ** 2 for one, two in zip(first, second, strict=True))


@lru_cache(maxsize=1024)
def to_xterm256(color: int) -> int:
    """Map 24-bit ``color`` to the nearest xterm cube or grayscale index."""
    rgb = _channels(color)
    cube = [
        min(range(6), key=lambda index: abs(_CUBE_LEVELS[index] - channel))
        for channel in rgb
    ]
    gray_index = round((sum(rgb) / 3 - _GRAY_START) / _GRAY_STEP)
    gray_index = min(_GRAY_LEVELS - 1, max(0, gray_index))
    gray = _GRAY_START + _GRAY_STEP * gray_index
    cube_rgb = tuple(_CUBE_LEVELS[i] for i in cube)
    if _distance((gray, gray, gray), rgb) < _distance(cube_rgb, rgb):
        return 232 + gray_index
    return 16 + 36 * cube[0] + 6 * cube[1] + cube[2]


@lru_cache(maxsize=1024)
def sgr(ink: Ink, depth: ColorDepth) -> str:
    """Return the SGR sequence switching from a reset state to ``ink`` at ``depth``."""
    codes = ["1"] * ink.bold + ["2"] * ink.faint
    if ink.color is not None and depth is ColorDepth.TRUE_COLOR:
        codes.append("38;2;{};{};{}".format(*_channels(ink.color)))
    elif ink.color is not None and depth is ColorDepth.EIGHT_BIT:
        codes.append(f"38;5;{to_xterm256(ink.color)}")
    return f"\x1b[{';'.join(codes)}m" if codes else ""


def to_plain(runs: Iterable[Run]) -> str:
    """Concatenate the text of ``runs`` without any styling."""
    return "".join([text for text, _ in runs])


def _coalesce(
    runs: Iterable[Run], look: Callable[[Ink], str]
) -> list[tuple[str, Ink, str]]:
    """Join neighboring runs whose inks ``look`` alike (spaces join any run).

    Example:
        >>> _coalesce([("1/2", FAINT), (" ", PLAIN), ("0:04", FAINT)], repr)[0][0]
        '1/2 0:04'
    """
    merged: list[tuple[str, Ink, str]] = []
    for text, ink in runs:
        if not text:
            continue
        if merged and not text.strip(" "):
            joined, kept, seen = merged[-1]
            merged[-1] = (joined + text, kept, seen)
            continue
        seen = look(ink)
        if merged and seen == merged[-1][2]:
            merged[-1] = (merged[-1][0] + text, merged[-1][1], seen)
        else:
            merged.append((text, ink, seen))
    return merged


@lru_cache(maxsize=4096)
def _switch(current: Ink, ink: Ink, depth: ColorDepth) -> str:
    same_weight = (ink.bold, ink.faint) == (current.bold, current.faint)
    keep = same_weight and ink.color is not None
    reset = "" if current == PLAIN or keep else "\x1b[0m"
    return reset + sgr(Ink(ink.color) if keep else ink, depth)


def to_ansi(runs: list[Run], depth: ColorDepth) -> str:
    """Serialize runs to ANSI, emitting SGR only when the style changes.

    Args:
        runs: Styled text runs.
        depth: The color capability of the target.

    Returns:
        Text with SGR sequences, always ending in the default style.
    """
    if depth is ColorDepth.NONE:
        return to_plain(runs)
    parts = []
    current = PLAIN
    for text, ink, _ in _coalesce(runs, lambda ink: sgr(ink, depth)):
        if ink != current:
            parts.append(_switch(current, ink, depth))
            current = ink
        parts.append(text)
    if current != PLAIN:
        parts.append("\x1b[0m")
    return "".join(parts)


@lru_cache(maxsize=1024)
def _css(ink: Ink, depth: ColorDepth) -> str:
    css = [f"color:#{ink.color:06x}"] if ink.color is not None and depth else []
    return ";".join(css + ["font-weight:bold"] * ink.bold + ["opacity:.6"] * ink.faint)


def to_html(runs: list[Run], depth: ColorDepth = ColorDepth.TRUE_COLOR) -> str:
    """Serialize styled runs to escaped HTML spans; ``depth`` NONE drops colors only."""
    parts = []
    for text, _, css in _coalesce(runs, lambda ink: _css(ink, depth)):
        escaped = html.escape(text, quote=False)
        parts.append(f'<span style="{css}">{escaped}</span>' if css else escaped)
    return "".join(parts)
