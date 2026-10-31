"""Flicker tests for the live terminal region.

Terminals without synchronized output show a frame as it arrives, so each
frame is replayed one character or escape sequence at a time, and no row that
shows text both before and after the frame may go blank in between.
"""

import logging
import re
import time
from collections.abc import Callable, Sequence

import pytest

from psnek import Bar, Options
from psnek._engine import LiveBackend
from psnek._render import log_lines
from tests.fakes import FakeTerminal, engine_for, engine_lock
from tests.vt import Screen

COLUMNS, ROWS = 40, 8
_STEP = re.compile(r"\x1b\[[0-9;?]*[@-~]|\x1b\][^\x1b]*\x1b\\|\x07|[^\x1b\x07]")


def _filled(screen: Screen) -> list[bool]:
    return ["".join(line).strip() != "" for line in screen.lines]


def blank_flashes(earlier: str, frame: str) -> list[int]:
    """Return the rows that go blank while ``frame`` is shown step by step.

    Args:
        earlier: Output already on the screen.
        frame: The frame to replay.

    Returns:
        Indexes of rows with text before and after the frame that were blank
        at some step of it.
    """
    screen = Screen(COLUMNS, ROWS)
    screen.feed(earlier)
    before = _filled(screen)
    steps = []
    for step in _STEP.findall(frame):
        screen.feed(step)
        steps.append(_filled(screen))
    after = steps[-1]
    steady = [row for row, shown in enumerate(before) if shown and after[row]]
    return [row for row in steady if not all(step[row] for step in steps)]


def _frames(
    terminal: FakeTerminal, actions: Sequence[Callable[[], object]]
) -> list[tuple[str, str]]:
    """Draw after each action, returning each frame with the output before it."""
    engine = engine_for(terminal)
    frames = []
    for action in actions:
        earlier = terminal.getvalue()
        with engine_lock():
            action()
            engine.draw(time.monotonic())
        frames.append((earlier, terminal.getvalue()[len(earlier) :]))
    return frames


@pytest.fixture
def sized(terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch) -> FakeTerminal:
    """The fake terminal at `COLUMNS` by `ROWS`."""
    monkeypatch.setenv("COLUMNS", str(COLUMNS))
    monkeypatch.setenv("LINES", str(ROWS))
    return terminal


def test_the_detector_sees_a_region_cleared_before_its_redraw() -> None:
    # Act
    flashes = blank_flashes("bar one\nbar two\n", "\x1b[2F\x1b[Jbar one\nbar two\n")

    # Assert
    assert flashes == [0, 1]


@pytest.mark.parametrize(
    "printed",
    [
        "x" * 5,
        "x" * (COLUMNS - 1),
        "x" * COLUMNS,
        "x" * (2 * COLUMNS + 3),
        "\u00e9" + "x" * (COLUMNS + 4),
        "",
    ],
)
def test_printing_above_the_bars_never_blanks_them(
    sized: FakeTerminal, printed: str
) -> None:
    # Arrange
    options = Options(stream=sized)
    with (
        Bar(total=10, desc="outer", options=options) as outer,
        Bar(total=10, desc="inner", options=options) as inner,
    ):
        # Act
        frames = _frames(
            sized,
            [
                outer.update,
                lambda: inner.write(printed),
                lambda: inner.log(printed or "logged", level=logging.WARNING),
                lambda: inner.write("\n".join(f"line {n}" for n in range(ROWS * 2))),
                inner.update,
            ],
        )
        screen = Screen(COLUMNS, ROWS)
        screen.feed(sized.getvalue())

    # Assert
    assert [blank_flashes(*frame) for frame in frames[1:]] == [[]] * 4
    assert "".join(screen.text()).count("x") == 2 * printed.count("x")


def test_finishing_a_kept_bar_never_blanks_the_others(sized: FakeTerminal) -> None:
    # Arrange
    options = Options(stream=sized, leave=True)
    with Bar(total=10, desc="outer", options=options) as outer:
        inner = Bar(total=1, desc="inner", options=options)

        # Act
        frames = _frames(sized, [outer.update, inner.close])

    # Assert
    assert blank_flashes(*frames[1]) == []


def test_a_progress_frame_rewrites_only_changed_rows(sized: FakeTerminal) -> None:
    # Arrange
    options = Options(stream=sized)
    with (
        Bar(total=10, desc="outer", options=options),
        Bar(total=10, desc="inner", options=options) as inner,
    ):
        # Act
        frames = _frames(sized, [inner.update, inner.update])

    # Assert: the unchanged outer row is skipped with a bare newline
    earlier, frame = frames[1]
    assert "outer" not in frame
    assert blank_flashes(earlier, frame) == []


def test_lines_written_over_the_bars_erase_all_but_their_last_cell() -> None:
    # Arrange: a wide character that does not fit wraps whole, ending its
    # row at the margin; a mark or a glyph a terminal draws at another width
    # may too, so those lines erase the region below them first instead
    backend = LiveBackend(FakeTerminal())
    tree = "\u251c\u2500"
    near = f"{tree} {'x' * (COLUMNS - 2 * len(tree) - 2)}"
    reaching = f"{tree} {'x' * (COLUMNS - 2 * len(tree) - 1)}"
    cases = [
        (near, COLUMNS, f"{near}\x1b[K\n"),
        (reaching, COLUMNS, f"\x1b[J{reaching}\n"),
        ("x" * 9 + "\u65e5" + "x" * 8, 10, "x" * 9 + "\u65e5" + "x" * 8 + "\n"),
        ("e\u0301" + "x" * 9, 10, "\x1b[Je\u0301" + "x" * 9 + "\n"),
        (
            "x" * 9 + "\u251c" + "x" * 8,
            10,
            "\x1b[J" + "x" * 9 + "\u251c" + "x" * 8 + "\n",
        ),
        ("\U0001f680", 80, "\U0001f680\x1b[K\n"),
    ]
    lead = len("  warn  ")
    styled = [
        log_lines(["x" * cells], logging.WARNING)[0] for cells in range(COLUMNS - lead)
    ]

    # Act
    written = [backend._over(line, columns) for line, columns, _ in cases]
    styled_written = [backend._over(line, COLUMNS) for line in styled]

    # Assert: escape sequences take no cells, so no styled line reaches it
    assert written == [expected for _, _, expected in cases]
    assert all(text.endswith("\x1b[K\n") for text in styled_written)
