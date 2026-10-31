"""Tests for the strict VT screen model used to check live output."""

import pytest

from tests.vt import Screen


def _feed(data: str, columns: int = 10, rows: int = 3) -> Screen:
    screen = Screen(columns, rows)
    screen.feed(data)
    return screen


def test_text_newlines_and_redraws() -> None:
    # Act
    screen = _feed("one\ntwo\n\x1b[2Funo\x1b[K\n\x1b[J")

    # Assert
    assert screen.text() == ["uno"]


def test_autowrap_wraps_or_rejects_overflow() -> None:
    assert _feed("x" * 12).text() == ["x" * 10, "xx"]
    with pytest.raises(ValueError, match="autowrap off"):
        _feed("\x1b[?7l" + "x" * 11)


def test_wide_zero_width_and_joiner_characters() -> None:
    # Act
    wide = _feed("日éx")
    joined = _feed("a\u200db\u200cc\u00add")

    # Assert
    assert wide.text() == ["日éx"]
    assert wide.column == 4
    assert joined.text() == ["a\u200db\u200cc\u00add"]
    assert joined.column == 5


def test_overwriting_half_of_a_wide_character_blanks_the_other_half() -> None:
    assert _feed("日\x1b[2Ga").text() == [" a"]
    assert _feed("a日\x1b[2Gb").text() == ["ab"]


@pytest.mark.parametrize(
    "data",
    [
        "backspace\x08",
        "\x1b]9;4;7;1\x1b\\",
        "csi\x9b2J",
        "\x1b[?1049h",
        "\x1b]52;c;eA==\x07",
        "rtl\u202e",
        "\x1b[38;2;300;0;0m",
    ],
)
def test_unexpected_output_is_rejected(data: str) -> None:
    with pytest.raises(ValueError, match=r"unexpected|control|range"):
        _feed(data)


def test_sgr_and_progress_osc_used_by_psnek_are_accepted() -> None:
    for data in ["\x1b[0;1;2;38;2;0;135;127m", "\x1b[38;5;30m", "\x1b]9;4;1;50\x1b\\"]:
        assert _feed(data + "ok").text() == ["ok"]


@pytest.mark.parametrize(
    "data",
    [
        "\x1b]9;job done: 3/3 0:40\x1b\\",
        "\x1b]777;notify;job failed;3/4 0:40 failed\x1b\\",
        "\x1b]99;i=psnek1:d=0:o=unfocused;job stuck\x1b\\",
        "\x1b]99;i=psnek1:p=body;3/4 0:40<0:13 stuck 48s\x1b\\",
        "\x07",
    ],
)
def test_notifications_and_bell_leave_the_screen_unchanged(data: str) -> None:
    assert _feed(f"ok{data}").text() == ["ok"]


def test_zero_width_without_a_base_cell_is_rejected() -> None:
    with pytest.raises(ValueError, match="base cell"):
        _feed("\x1b[10G\u0301")


def test_cursor_may_reach_the_top_row_but_not_scrollback() -> None:
    assert _feed("a\nb\nc\n\x1b[2Fx").text() == ["a", "x", "c"]
    with pytest.raises(ValueError, match="above the visible screen"):
        _feed("a\nb\nc\nd\n\x1b[4F")
    screen = _feed("a\nb\nc\nd\ne\nf\n\x1b[1F\x1b[J")
    with pytest.raises(ValueError, match="above the visible screen"):
        screen.feed("\x1b[2F")


def test_zero_and_oversized_column_parameters_follow_vt_rules() -> None:
    assert _feed("ab\x1b[0Gx").text() == ["xb"]
    assert _feed("ab\x1b[99Gx").text() == ["ab       x"]


def test_cells_keep_the_pen_they_were_written_with() -> None:
    # Act
    screen = _feed("\x1b[1;38;2;1;2;3ma\x1b[0m \x1b[2mb\x1b[38;5;9mc\x1b[0md\x1b[K")

    # Assert
    assert screen.styled() == [
        [
            ("a", (True, False, "2;1;2;3")),
            (" ", (False, False, "")),
            ("b", (False, True, "")),
            ("c", (False, True, "5;9")),
            ("d", (False, False, "")),
        ]
    ]


def test_erasing_drops_the_pens_of_erased_cells() -> None:
    # Act
    screen = _feed("\x1b[1mabc\x1b[0m\x1b[2G\x1b[K")

    # Assert
    assert screen.styled() == [[("a", (True, False, ""))]]
