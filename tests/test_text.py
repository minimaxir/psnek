"""Tests for sanitization, display width, and number/time formatting."""

from typing import cast

from psnek._text import (
    Scale,
    char_width,
    display_width,
    disputed,
    encodable,
    format_age,
    format_duration,
    format_number,
    format_rate,
    pad,
    sanitize,
    truncate,
    width_bounds,
    wrap,
)


def test_sanitize_neutralizes_control_sequences() -> None:
    for raw, expected in [
        ("plain text", "plain text"),
        ("\x1b[31mred\x1b[0m", "red"),
        ("\x1b]52;c;ZXZpbA==\x07copied", "copied"),
        ("\x1b]8;;https://evil\x1b\\link\x1b]8;;\x1b\\", "link"),
        ("a\tb", "a       b"),
        ("\u65e5\u672c\tx", "\u65e5\u672c    x"),
        ("e\u0301\tx", "e\u0301       x"),
        ("\x07\tx", "\\u0007  x"),
        ("bell\x07", "bell\\u0007"),
        ("csi\x9b2J", "csi\\u009b2J"),
        ("rtl\u202eoverride", "rtl\\u202eoverride"),
        ("surrogate\udc80", "surrogate\\udc80"),
        ("tag\U000e0041", "tag\\U000e0041"),
        ("open\x1b]52;c;never terminated", "open52;c;never terminated"),
        ("no\u00a0break\u3000wide", "no break  wide"),
        ("\x1b[1mbold\x1b[0m\tcol", "bold    col"),
        (
            "\U0001f468\u200d\U0001f469 \u0645\u06cc\u200c co\u00adop",
            "\U0001f468\u200d\U0001f469 \u0645\u06cc\u200c co\u00adop",
        ),
        ("x" * 100_000, "x" * 100_000),
    ]:
        result = sanitize(raw)
        assert result == expected
        assert "\x1b" not in result


def test_display_width_counts_cells() -> None:
    for text, width in [
        ("abc", 3),
        ("日本語", 6),
        ("é", 1),
        ("e\u0301", 1),
        ("\u28ff\u2800", 2),
        ("\u2764\ufe0f", 2),
        ("\U0001f1fa\U0001f1f8", 2),
        ("\U0001f5a5\ufe0f", 2),
        ("x\U0001f5a5\ufe0fy", 4),
    ]:
        assert display_width(text) == width
    assert char_width("\u0301") == char_width("\u200d") == char_width("\u200c") == 0


def test_box_drawing_and_braille_take_one_cell_each() -> None:
    # Arrange
    blocks = (range(0x2500, 0x2580), range(0x2800, 0x2900))
    narrow = [chr(code) for block in blocks for code in block]

    # Act
    widths = set(map(char_width, narrow))
    mixed = display_width("".join(narrow) + "ab")

    # Assert
    assert widths == {1}
    assert mixed == len(narrow) + 2


def test_truncate_respects_cells() -> None:
    for text, width, expected in [
        ("hello", 10, "hello"),
        ("hello world", 8, "hello..."),
        ("日本語テキスト", 7, "日本..."),
        ("abc", 2, ".."),
        ("日本", 1, "."),
        ("日本語", 6, "日本語"),
        ("\U0001f5a5\ufe0f box", 6, "\U0001f5a5\ufe0f box"),
        ("\U0001f5a5\ufe0f box", 5, "\U0001f5a5\ufe0f..."),
        ("\u65e5\ufe0f\ufe0f", 2, "\u65e5\ufe0f\ufe0f"),
        ("\u65e5\u0301\ufe0f", 2, "\u65e5\u0301\ufe0f"),
    ]:
        result = truncate(text, width)
        assert result == expected
        assert display_width(result) <= width


def test_wrap_breaks_at_spaces_and_ends_an_overflow_in_an_ellipsis() -> None:
    for text, width, rows, expected in [
        ("", 8, 2, ()),
        ("one two", 8, 2, ("one two",)),
        ("one two three", 8, 2, ("one two", "three")),
        ("one   two\tthree", 8, 3, ("one two", "three")),
        ("one two three four", 8, 2, ("one two", "three...")),
        ("abcdefghijklmnop", 6, 3, ("abcdef", "ghijkl", "mnop")),
        ("abcdefghijklmnopqrs", 6, 3, ("abcdef", "ghijkl", "mno...")),
        ("日本語の説明です", 5, 2, ("日本", "語...")),
        ("a b", 6, 1, ("a b",)),
    ]:
        result = wrap(text, width, rows)
        assert result == expected
        assert len(result) <= rows
        assert all(display_width(line) <= width for line in result)


def test_wrap_stops_measuring_after_the_last_row() -> None:
    # Arrange
    text = "word " * 1_000_000

    # Act
    result = wrap(text, 10, 2)

    # Assert
    assert result == ("word word", "word wo...")


def test_long_runs_of_combining_marks_are_cut() -> None:
    # Arrange
    line = "a" + "\u0301" * 1_000_000

    # Act
    result = truncate(line, 10)

    # Assert
    assert len(result) < 20
    assert result.endswith("...")


def test_encodable_escapes_what_a_codec_cannot_write() -> None:
    assert encodable("caf\u00e9", "ascii") == "caf\\xe9"


def test_truncating_a_huge_line_measures_only_its_start() -> None:
    # Arrange
    line = "日" * 2_000_000

    # Act
    result = truncate(line, 10)

    # Assert
    assert result == "日日日..."


def test_format_number() -> None:
    for value, scale, expected in [
        (42, "none", "42"),
        (2.5, "none", "2.50"),
        (999, "si", "999"),
        (1234, "si", "1.23k"),
        (12345, "si", "12.3k"),
        (999_999, "si", "1.00M"),
        (1536, "iec", "1.50Ki"),
        (5 * 2**30, "iec", "5.00Gi"),
        (999.4, "si", "999"),
        (999.5, "si", "1.00k"),
        (1023, "iec", "1023"),
        (1023.5, "iec", "1.00Ki"),
        (1024**2 - 512, "iec", "1.00Mi"),
    ]:
        assert format_number(value, cast("Scale", scale)) == expected


def test_format_duration() -> None:
    for seconds, expected in [
        (None, "--:--"),
        (float("inf"), "--:--"),
        (-1.0, "--:--"),
        (5.9, "0:05"),
        (3599, "59:59"),
        (3600, "1:00:00"),
        (90061, "1d01h"),
        (99 * 86400 + 86399, "99d23h"),
        (100 * 86400, "99d+"),
        (float("nan"), "--:--"),
    ]:
        assert format_duration(seconds) == expected


def test_format_rate() -> None:
    for rate, scale, expected in [
        (None, "none", "--it/s"),
        (0.0, "none", "--it/s"),
        (-2.0, "none", "--it/s"),
        (float("inf"), "none", "--it/s"),
        (float("nan"), "none", "--it/s"),
        (0.25, "none", "4.00s/it"),
        (1 / 600, "none", "10.0min/it"),
        (1 / 7200, "none", "2.00h/it"),
        (1 / 3_600_000, "none", ">999h/it"),
        (5e-324, "none", ">999h/it"),
        (12.34, "none", "12.3it/s"),
        (1234.0, "none", "1234it/s"),
        (54321.0, "none", "54.3kit/s"),
        (16.4 * 2**20, "iec", "16.4MiB/s"),
        (1 / 59.96, "none", "1.00min/it"),
        (1 / 3599.7, "none", "1.00h/it"),
        (1 / 3_598_560, "none", ">999h/it"),
        (99.96, "none", "100it/s"),
        (9999.7, "none", "10.0kit/s"),
    ]:
        unit = "B" if scale == "iec" else "it"
        assert format_rate(rate, unit, cast("Scale", scale)) == expected


def test_format_age() -> None:
    for seconds, expected in [
        (0, "0s"),
        (59.9, "59s"),
        (60, "1m00s"),
        (3599, "59m59s"),
        (90061, "25h01m"),
        (float("inf"), "--"),
        (float("nan"), "--"),
    ]:
        assert format_age(seconds) == expected


def test_pad_counts_display_cells() -> None:
    assert pad("画像", 6) == "  画像"
    assert pad("ab", 4, left=False) == "ab  "
    assert pad("abcdef", 4) == "abcdef"


def test_huge_values_are_capped() -> None:
    for text, expected in [
        (format_age(359_999), "99h59m"),
        (format_age(360_000), "99h+"),
        (format_age(1e308), "99h+"),
        (format_number(9.99e26, "si"), "999Y"),
        (format_number(1e308, "si"), ">999Y"),
        (format_number(-1e308, "iec"), "<-999Yi"),
        (format_rate(1e308, "it"), ">999Yit/s"),
    ]:
        assert text == expected


def test_widths_terminals_may_draw_differently_are_told_and_bounded() -> None:
    # Arrange: ASCII, Braille and common CJK text are agreed everywhere
    agreed = [
        "plain ascii 123",
        "\u28ff\u2840 braille",
        "\u65e5\u672c\u3002\ud55c\u3041\u30fe\uff01\u3400\u4dbf\u300c\u300f",
    ]
    unsure = ["\u251c\u2500 tree", "caf\u00e9", "\U0001f468\u200d\U0001f469"]
    unsure += ["\uff76\uff9e", "\u3099", "\u302a", "\u303f"]
    bounded = ["plain", "\u65e5\u672c", "\u251c\u2500 x", "e\u0301", "\U0001f680"]

    # Act
    told = [disputed(text) for text in agreed + unsure]
    bounds = [width_bounds(text) for text in bounded]

    # Assert
    assert told == [False] * len(agreed) + [True] * len(unsure)
    assert bounds == [(5, 5), (4, 4), (2, 6), (1, 3), (0, 2)]
