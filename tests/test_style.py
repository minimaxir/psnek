"""Tests for color math, accessibility, and run serialization."""

from dataclasses import replace

import pytest

from psnek._frame import Frame
from psnek._glyphs import ASCII, Style
from psnek._render import GALLERY, Columns, layout, shimmer
from psnek._style import (
    BOLD,
    DEFAULT_PALETTE,
    FAINT,
    PLAIN,
    ColorDepth,
    Ink,
    blend,
    gradient_at,
    parse_hex,
    sgr,
    to_ansi,
    to_html,
    to_plain,
    to_xterm256,
)


def _luminance(color: int) -> float:
    def linear(channel: int) -> float:
        value = channel / 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (color >> 16) & 255, (color >> 8) & 255, color & 255
    return 0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)


def _contrast(first: int, second: int) -> float:
    low, high = sorted((_luminance(first), _luminance(second)))
    return (high + 0.05) / (low + 0.05)


# Black, GitHub dark, VS Code dark, One Dark, Dracula, Solarized dark, white,
# GitHub light, and Solarized light.
_BACKGROUNDS = (
    0x000000, 0x0D1117, 0x1E1E1E, 0x282C34, 0x282A36, 0x002B36,
    0xFFFFFF, 0xF6F8FA, 0xFDF6E3,
)  # fmt: skip
_CUBE = (0, 95, 135, 175, 215, 255)


def _xterm_rgb(index: int) -> int:
    if index >= 232:
        level = 8 + 10 * (index - 232)
        return level << 16 | level << 8 | level
    index -= 16
    return _CUBE[index // 36] << 16 | _CUBE[index // 6 % 6] << 8 | _CUBE[index % 6]


def _emitted_colors() -> set[int]:
    colors: set[int] = set()
    for style in (Style(), Style(glyphs=ASCII), Style(depth=ColorDepth.EIGHT_BIT)):
        for _, frame in GALLERY:
            for step in range(40):
                for width in (40, 100):
                    runs = layout(
                        replace(frame, phase=step / 13), width, style, Columns(12)
                    )[0]
                    colors |= {ink.color for _, ink in runs if ink.color is not None}
    palette = DEFAULT_PALETTE
    bases = [gradient_at(palette.gradient, i / 100) for i in range(101)]
    bases += [palette.stalled, palette.failed, palette.interrupted, palette.shimmer]
    levels = [i / 8 for i in range(9)]
    colors |= {shimmer(level, palette, base) for base in bases for level in levels}
    return colors


@pytest.mark.parametrize("depth", [ColorDepth.TRUE_COLOR, ColorDepth.EIGHT_BIT])
def test_every_emitted_color_meets_non_text_contrast(depth: ColorDepth) -> None:
    # Arrange
    colors = sorted(_emitted_colors())
    if depth is ColorDepth.EIGHT_BIT:
        colors = [_xterm_rgb(to_xterm256(color)) for color in colors]

    # Act
    worst = min(
        _contrast(color, background) for color in colors for background in _BACKGROUNDS
    )

    # Assert
    assert len(colors) > 10
    assert worst >= 3.0


def test_a_running_comet_head_stands_out_in_256_colors() -> None:
    # Arrange
    style = Style(depth=ColorDepth.EIGHT_BIT)
    frames = [Frame(phase=step / 13) for step in range(40)]

    # Act
    rows = [layout(frame, 40, style)[0] for frame in frames]

    # Assert
    for runs in rows:
        body = {to_xterm256(ink.color) for _, ink in runs if ink.color is not None}
        assert len(body) == 2


def test_blend_gradient_and_xterm256_mapping() -> None:
    for color, index in (
        (0xFF0000, 196),
        (0x00FF00, 46),
        (0x000000, 16),
        (0x808080, 244),
        (0xFFFFFF, 231),
    ):
        assert to_xterm256(color) == index
    assert blend(0x000000, 0xFFFFFF, 0.5) == 0x808080
    assert gradient_at((0x112233,), 0.7) == 0x112233
    assert gradient_at(DEFAULT_PALETTE.gradient, 0.0) == DEFAULT_PALETTE.gradient[0]
    assert gradient_at(DEFAULT_PALETTE.gradient, 1.0) == DEFAULT_PALETTE.gradient[-1]


def test_parse_hex_reads_six_digit_colors_with_an_optional_hash() -> None:
    assert parse_hex(("#00948B", "ff0000", "#abcdef")) == (0x00948B, 0xFF0000, 0xABCDEF)


@pytest.mark.parametrize(
    "colors", [(), ("#fff",), ("#12345g",), ("0x1234",), ("#ff0000 ",), ("##ff0000",)]
)
def test_parse_hex_rejects_anything_but_six_hex_digits(
    colors: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="RRGGBB"):
        parse_hex(colors)


def test_sgr_by_depth() -> None:
    ink = Ink(0x0E9E8A, bold=True)
    assert sgr(ink, ColorDepth.TRUE_COLOR) == "\x1b[1;38;2;14;158;138m"
    assert sgr(ink, ColorDepth.EIGHT_BIT).startswith("\x1b[1;38;5;")
    assert sgr(PLAIN, ColorDepth.TRUE_COLOR) == ""


def test_to_ansi_avoids_redundant_resets() -> None:
    # Arrange
    runs = [("a", Ink(0x111111)), ("b", Ink(0x222222)), ("c", PLAIN), ("d", BOLD)]

    # Act
    text = to_ansi(runs, ColorDepth.TRUE_COLOR)

    # Assert
    assert text == ("\x1b[38;2;17;17;17ma\x1b[38;2;34;34;34mb\x1b[0mc\x1b[1md\x1b[0m")

    plain_runs = [("x", BOLD), ("y", FAINT)]
    assert to_ansi(plain_runs, ColorDepth.NONE) == to_plain(plain_runs) == "xy"


def test_to_html_escapes_and_styles() -> None:
    # Act
    markup = to_html([("<b>&", Ink(0xABCDEF, bold=True, faint=True)), ("'q'", PLAIN)])

    # Assert
    assert markup == (
        '<span style="color:#abcdef;font-weight:bold;opacity:.6">&lt;b&gt;&amp;</span>'
        "'q'"
    )


def test_serializers_join_runs_that_render_alike() -> None:
    # Arrange: spaces show no color or weight, so they join the run before
    # them whatever their style; inks rendering alike at a depth join too
    spaced = [("1/2", FAINT), (" ", PLAIN), ("0:04", FAINT), ("  ", BOLD), ("x", PLAIN)]
    same_weight = [("a", Ink(0xABCDEF, bold=True)), ("b", Ink(0x123456, bold=True))]
    one_index = [("a", Ink(0x000001)), ("b", Ink(0x000002))]
    empty = [("", BOLD), ("x", PLAIN), ("", FAINT)]

    # Act
    written = [
        to_html(spaced),
        to_ansi(spaced, ColorDepth.TRUE_COLOR),
        to_html([("ok", BOLD), (" ", FAINT), ("go", BOLD)]),
        to_html(same_weight, ColorDepth.NONE),
        to_ansi(one_index, ColorDepth.EIGHT_BIT),
        to_ansi(empty, ColorDepth.EIGHT_BIT),
        to_html(empty),
    ]

    # Assert
    assert written == [
        '<span style="opacity:.6">1/2 0:04  </span>x',
        "\x1b[2m1/2 0:04  \x1b[0mx",
        '<span style="font-weight:bold">ok go</span>',
        '<span style="font-weight:bold">ab</span>',
        "\x1b[38;5;16mab\x1b[0m",
        "x",
        "x",
    ]


def test_to_html_without_color_keeps_weight() -> None:
    # Act
    markup = to_html(
        [("a", Ink(0xABCDEF, bold=True)), ("b", Ink(0x123456))], ColorDepth.NONE
    )

    # Assert
    assert markup == '<span style="font-weight:bold">a</span>b'
