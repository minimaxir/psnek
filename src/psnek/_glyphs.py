"""Glyph sets, from Braille cells with kaomoji comet heads to ASCII, and styles."""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

from psnek._style import DEFAULT_PALETTE, ColorDepth, Palette
from psnek._text import display_width, is_text_pair

FILL_ORDER = (7, 3, 2, 1, 4, 5, 6, 8)
STEPS = len(FILL_ORDER)
INDENT, _SPINNER_LENGTH, _TREE_PARTS = 2, 3, 4
_DOT_BITS = frozenset(1 << dot for dot in range(STEPS))
_PERIMETER = (1, 4, 5, 6, 8, 7, 3, 2)
_CELL_DOTS = ((1, 2, 3, 7), (4, 5, 6, 8))


Dots = set[tuple[int, int]]


def _braille(*dots: int) -> str:
    return chr(0x2800 + sum(1 << (dot - 1) for dot in dots))


def _blob(left: int, top: int, size: int, hollow: bool = False) -> Dots:
    radius = size / 2
    inner = (radius - 1) ** 2 if hollow and radius > 1 else -1.0
    return {
        (left + x, top + y)
        for x in range(size)
        for y in range(size)
        if inner < (x + 0.5 - radius) ** 2 + (y + 0.5 - radius) ** 2 <= radius**2
    }


def _smile(height: int) -> Dots:
    """Draw small eyes over a U-shaped smile whose bottom rests on the baseline."""
    wide, tall, edge = 4 * height + 2, 4 * height, 4 * height + 1
    rows = [
        tall - 1 - round((2 * height - 1) * (abs(2 * x - edge) / edge) ** 4)
        for x in range(wide // 2 + 1)
    ]
    mouth = {
        (x, y)
        for x in range(wide // 2)
        for y in range(rows[x], max(rows[x], rows[x + 1] - 1) + 1)
    }
    return _blob(height, 0, height) | mouth


def _worry(height: int) -> Dots:
    """Draw wide round eyes over a smaller open mouth on the baseline."""
    mouth = 2 * ((height + 2) // 2)
    left, top = 2 * height + 1 - mouth // 2, 4 * height - mouth
    return _blob(0, 0, 2 * height, hollow=True) | _blob(left, top, mouth, hollow=True)


def _wince(height: int) -> Dots:
    """Draw ``>``-shaped squeezed eyes over a flat mouth on the baseline."""
    eye = {(step, step) for step in range(height + 1)}
    eye |= {(height - step, height + step) for step in range(height + 1)}
    return eye | {(x, 4 * height - 1) for x in range(height + 1, 3 * height + 1)}


@lru_cache(maxsize=32)
def _face(draw: Callable[[int], Dots], height: int) -> tuple[str, ...]:
    """Render the left half of a face from ``draw``, mirrored, as Braille rows."""
    half = draw(height)
    wide = 4 * height + 2
    masks = [[0] * (wide // 2) for _ in range(height)]
    for x, y in half | {(wide - 1 - x, y) for x, y in half}:
        masks[y // 4][x // 2] |= 1 << (_CELL_DOTS[x % 2][y % 4] - 1)
    return tuple("".join(chr(0x2800 | mask) for mask in row) for row in masks)


# Kaomoji drawn at every bar height, which at one row fill a whole cell with
# each eye and the mouth where they can, so the gaps fonts leave between Braille
# cells do not break them up: a smile resting on the baseline the snake crawls
# on, wide worried eyes, and a wince.
_FACES: dict[str, Callable[[int], Dots]] = {
    _face(draw, 1)[0]: draw for draw in (_smile, _worry, _wince)
}
_SMILE, _WORRY, _WINCE = _FACES


@lru_cache(maxsize=32)
def tall_head(head: str, height: int) -> tuple[str, ...]:
    """Return comet ``head`` redrawn ``height`` rows tall (faces) or repeated."""
    draw = _FACES.get(head)
    return _face(draw, height) if draw else (head,) * height


@dataclass(frozen=True, slots=True)
class Glyphs:
    """Characters used to draw a bar.

    Attributes:
        partials: Cell glyphs indexed by filled eighths (``0``..``8``).
        track: Glyph for unfilled cells.
        spinner: Animation frames for the activity indicator.
        done: Indicator once finished successfully.
        failed: Indicator after an exception.
        interrupted: Indicator when closed before completion.
        gutter: Prefix drawn before every panel line.
        dot_bits: Bit for each fill step, used to draw the indeterminate comet.
        comet_heads: Heads drawn ahead of the comet moving right and left, if any.
        stalled_heads: Heads drawn in their place while stalled.
        stuck_heads: Heads drawn in their place while stuck.
        tree: Two-cell guide, blank, branch, and last-branch prefixes of nested bars.

    Raises:
        ValueError: If the glyphs cannot draw every state.
    """

    partials: tuple[str, ...]
    track: str
    spinner: tuple[str, ...]
    done: str
    failed: str
    interrupted: str
    gutter: str
    dot_bits: tuple[int, ...] = ()
    comet_heads: tuple[str, str] | None = None
    stalled_heads: tuple[str, str] | None = None
    stuck_heads: tuple[str, str] | None = None
    tree: tuple[str, str, str, str] = ("\u2502 ", "  ", "\u251c\u2500", "\u2514\u2500")

    def __post_init__(self) -> None:
        """Reject glyph sets that cannot draw every state."""
        if len(self.partials) != STEPS + 1 or not self.spinner:
            raise ValueError("glyphs need nine partials and at least one spinner")
        bits = self.dot_bits
        if bits and not (len(bits) == len(set(bits)) == STEPS and {*bits} <= _DOT_BITS):
            raise ValueError("glyph dot_bits need a distinct Braille dot per fill step")
        pairs = (self.comet_heads, self.stalled_heads, self.stuck_heads)
        heads = [pair for pair in pairs if pair is not None]
        if heads and (self.comet_heads is None or not all(map(is_text_pair, heads))):
            raise ValueError("glyph comet heads need strings of one nonzero width")
        widths = {display_width(head) for pair in heads for head in pair}
        if len(widths) > 1 or 0 in widths:
            raise ValueError("glyph comet heads need strings of one nonzero width")
        if len(self.tree) != _TREE_PARTS or any(
            not isinstance(part, str) or display_width(part) != INDENT
            for part in self.tree
        ):
            raise ValueError("glyph tree needs four two-cell strings")


BRAILLE = Glyphs(
    partials=tuple(_braille(*FILL_ORDER[:step]) for step in range(9)),
    track=_braille(7, 8),
    spinner=tuple(
        _braille(*(_PERIMETER[(start + i) % 8] for i in range(_SPINNER_LENGTH)))
        for start in range(8)
    ),
    done=_braille(*range(1, 9)),
    failed="x",
    interrupted="!",
    gutter=_braille(1, 2, 3, 7) + " ",
    dot_bits=tuple(1 << (dot - 1) for dot in FILL_ORDER),
    comet_heads=(_SMILE, _SMILE),
    stalled_heads=(_WORRY, _WORRY),
    stuck_heads=(_WINCE, _WINCE),
)
ASCII = Glyphs(
    partials=(" ", ".", ":", "-", "=", "+", "*", "%", "#"),
    track="_",
    spinner=("|", "/", "-", "\\"),
    done="#",
    failed="x",
    interrupted="!",
    gutter="| ",
    comet_heads=(">", "<"),
    tree=("| ", "  ", "|-", "`-"),
)


@dataclass(frozen=True, slots=True)
class Style:
    """Output styling decisions shared by every frame on one target.

    Attributes:
        glyphs: Character set.
        palette: Colors.
        depth: Color capability of the target.

    Raises:
        TypeError: If ``glyphs`` is not a `Glyphs` or ``palette`` not a `Palette`.
        ValueError: If ``depth`` is not a `ColorDepth` value.
    """

    glyphs: Glyphs = BRAILLE
    palette: Palette = field(default=DEFAULT_PALETTE)
    depth: ColorDepth = ColorDepth.TRUE_COLOR

    def __post_init__(self) -> None:
        """Check the field types and normalize ``depth`` to a `ColorDepth`."""
        if not isinstance(self.glyphs, Glyphs) or not isinstance(self.palette, Palette):
            raise TypeError("a Style needs Glyphs and a Palette")
        object.__setattr__(self, "depth", ColorDepth(self.depth))
