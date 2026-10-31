"""Tests for panel content laid out as text lines and borderless tables."""

import sys
from collections import namedtuple
from dataclasses import dataclass, field

import pytest

from psnek._style import BOLD, FAINT, PLAIN, PanelLine, to_plain
from psnek._table import CELL_WIDTH, Justify, panel_lines


def _texts(lines: list[PanelLine]) -> list[str]:
    return [line if isinstance(line, str) else to_plain(line) for line in lines]


@dataclass
class _Worker:
    name: str
    items: int
    secret: str = field(default="hidden", repr=False)


_Point = namedtuple("_Point", "x y")


def test_mapping_shows_bold_keys_beside_formatted_values() -> None:
    # Act
    lines = panel_lines({"loss": 0.1234567, "learning_rate": 3e-4, "grad": None}, 8)

    # Assert
    assert _texts(lines) == [
        "loss           0.1235",
        "learning_rate  0.0003000",
        "grad           None",
    ]
    assert lines[0] == (("loss           ", BOLD), ("0.1235", PLAIN))
    assert lines[2][1] == ("None", FAINT)


def test_records_get_a_bold_header_in_first_seen_key_order() -> None:
    # Arrange
    records = [{"model": "llama", "acc": 0.91}, {"model": "gpt2", "f1": 0.5}]

    # Act
    lines = panel_lines(records, 8)

    # Assert
    assert _texts(lines) == [
        "model  acc     f1",
        "llama  0.9100",
        "gpt2           0.5000",
    ]
    assert lines[0] == (("model  acc     f1", BOLD),)
    assert not any(text.endswith(" ") for text in _texts(lines))


@pytest.mark.parametrize(
    ("justify", "expected"),
    [
        ("left", ["name  n", "a     1", "bbbb  222"]),
        ("right", ["name    n", "   a    1", "bbbb  222"]),
        ("center", ["name   n", " a     1", "bbbb  222"]),
    ],
)
def test_justification_aligns_every_column(
    justify: Justify, expected: list[str]
) -> None:
    # Act
    lines = panel_lines([{"name": "a", "n": 1}, {"name": "bbbb", "n": 222}], 8, justify)

    # Assert
    assert _texts(lines) == expected


def test_dataclasses_and_named_tuples_are_records() -> None:
    # Act
    single = panel_lines(_Worker("w1", 3), 8)
    many = panel_lines([_Worker("w1", 3), _Worker("w22", 40)], 8)
    point = panel_lines(_Point(1, 2), 8)

    # Assert: fields hidden from repr stay hidden
    assert _texts(single) == ["name   w1", "items  3"]
    assert _texts(many) == ["name  items", "w1    3", "w22   40"]
    assert _texts(point) == ["x  1", "y  2"]


def test_other_sequences_become_rows_without_a_header() -> None:
    # Act
    rows = panel_lines([("shard-1", "done", 1200), ("shard-22", "running")], 8)
    items = panel_lines(["first", "second"], 8)
    mixed = panel_lines([{"a": 1}, "loose"], 8)

    # Assert
    assert _texts(rows) == ["shard-1   done     1200", "shard-22  running"]
    assert rows[1] == (("shard-22  running", PLAIN),)
    assert _texts(items) == ["first", "second"]
    assert _texts(mixed) == ["{'a': 1}", "loose"]


def test_overflow_formats_only_shown_rows_and_counts_the_rest() -> None:
    # Arrange
    formatted: list[int] = []

    class Counted:
        def __init__(self, index: int) -> None:
            self.index = index

        def __str__(self) -> str:
            formatted.append(self.index)
            return f"item {self.index}"

    # Act
    records = panel_lines([{"i": i} for i in range(20)], 4)
    mapping = panel_lines({f"k{i}": i for i in range(10)}, 3)
    items = panel_lines([Counted(i) for i in range(1000)], 3)

    # Assert
    assert _texts(records) == ["i", "0", "1", "... 18 more rows"]
    assert records[-1] == (("... 18 more rows", FAINT),)
    assert _texts(mapping) == ["k0  0", "k1  1", "... 8 more rows"]
    assert _texts(items) == ["item 0", "item 1", "... 998 more rows"]
    assert formatted == [0, 1]


def test_a_header_needs_room_for_a_row() -> None:
    # Act
    lines = panel_lines([{"i": 1}, {"i": 2}], 1)

    # Assert
    assert _texts(lines) == ["... 2 more rows"]


def test_text_is_split_sanitized_and_elided() -> None:
    # Act
    lines = panel_lines("a\x1b[31mred\nb\nc", 2)
    number = panel_lines(42, 2)

    # Assert
    assert lines == ["ared", (("... 2 more lines", FAINT),)]
    assert number == ["42"]


@pytest.mark.parametrize("content", [None, "", {}, [], ()])
def test_empty_content_clears(content: object) -> None:
    assert panel_lines(content, 8) == []


def test_no_lines_without_room() -> None:
    assert panel_lines({"a": 1}, 0) == []


def test_cells_are_sanitized_and_cut_except_the_last_column() -> None:
    # Arrange
    long = "x" * (CELL_WIDTH + 10)

    # Act
    lines = panel_lines({"\x1b]52;c;evil\x07key": long, long: "\tv"}, 8)

    # Assert
    texts = _texts(lines)
    assert "\x1b" not in "".join(texts)
    assert texts[0].startswith("key")
    assert texts[0].endswith(long)
    assert texts[1].startswith("x" * (CELL_WIDTH - 3) + "...")
    assert texts[1].endswith("  v")


def test_polars_frames_fetch_only_the_rows_shown() -> None:
    # Arrange
    polars = pytest.importorskip("polars")
    frame = polars.DataFrame({"model": [f"m{i}" for i in range(9)], "acc": None})

    # Act
    lines = panel_lines(frame, 4)

    # Assert
    assert _texts(lines) == [
        "model  acc",
        "m0     None",
        "m1     None",
        "... 7 more rows",
    ]
    assert lines[1][-1] == ("None", FAINT)


def test_pandas_frames_show_a_named_index_but_not_the_default_one() -> None:
    # Arrange
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame({"acc": [0.91234, 0.5], "n": [3, 40]})

    # Act
    plain = panel_lines(frame, 8)
    indexed = panel_lines(frame.set_axis(["llama", "gpt2"]).rename_axis("model"), 8)

    # Assert
    assert _texts(plain) == ["acc     n", "0.9123  3", "0.5000  40"]
    assert _texts(indexed) == [
        "model  acc     n",
        "llama  0.9123  3",
        "gpt2   0.5000  40",
    ]


def test_frames_are_not_recognized_without_their_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    polars = pytest.importorskip("polars")
    frame = polars.DataFrame({"a": [1]})
    monkeypatch.delitem(sys.modules, "polars")

    # Act
    lines = panel_lines(frame, 8)

    # Assert: never imported by psnek, the frame falls back to its str
    assert _texts(lines) == str(frame).split("\n")
    assert "polars" not in sys.modules
