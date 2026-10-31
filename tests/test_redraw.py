"""Tests that redrawing only changed runs leaves exactly what a full redraw would.

Frames of mixed activity are replayed into the VT model, which records every
cell's text and pen; after each frame, the live region must match a fresh
screen given every row in full. Terminals that draw some characters at other
widths than psnek measures are modeled too.
"""

import random
import time
import unicodedata
from collections.abc import Callable
from itertools import product

import pytest

from psnek import Bar, Options
from psnek._backends import Backend
from psnek._bar import PanelPosition
from psnek._engine import LiveBackend
from psnek._redraw import SYNC_BEGIN, SYNC_END, WRAP_OFF
from psnek._redraw import drawn as _drawn
from psnek._style import BOLD, PLAIN, Ink, Run
from psnek._text import char_width
from tests.fakes import FakeTerminal, FlakyTerminal, engine_for, engine_lock
from tests.vt import Pen, Screen

COLUMNS, ROWS = 60, 12
DESCRIPTIONS = (
    "load",
    "load 日本語 shards",
    "emoji \U0001f680 run",
    "\U0001f321 temperature",
    "team \U0001f468\u200d\U0001f469\u200d\U0001f467 sync",
    "a much longer name",
)

Cells = list[list[tuple[str, Pen]]]


def _ambiguous_wide(char: str) -> int:
    """Draw East Asian ambiguous characters two cells wide, as CJK setups do."""
    if unicodedata.east_asian_width(char) == "A":
        return 2
    return char_width(char)


def _pictographs_narrow(char: str) -> int:
    """Draw pictographs without emoji presentation one cell wide, as wcwidth does."""
    if ord(char) >= 0x1F300 and unicodedata.east_asian_width(char) == "N":
        return 1
    return char_width(char)


def _joined_whole(char: str) -> int:
    """Draw the family in `DESCRIPTIONS` as one two-cell cluster, as many do."""
    return 0 if char in "\U0001f469\U0001f467" else char_width(char)


WIDTHS = {
    "psnek's": char_width,
    "ambiguous wide": _ambiguous_wide,
    "pictographs narrow": _pictographs_narrow,
    "joined whole": _joined_whole,
}


def _screen(width: Callable[[str], int]) -> Screen:
    """A screen drawing with ``width``, with room for rows it draws wider.

    A row drawn wider than the terminal would clip at its margin, which the
    model refuses with autowrap off, so clipping is not modeled here.
    """
    return Screen(COLUMNS if width is char_width else 2 * COLUMNS, ROWS, width)


def _full_redraw(
    backend: LiveBackend, width: Callable[[str], int] = char_width
) -> Cells:
    screen = _screen(width)
    rows = backend.region
    screen.feed(WRAP_OFF + "".join(f"{backend._line(row.runs)}\n" for row in rows))
    return screen.styled(0, len(rows))


def _live_region(screen: Screen, backend: LiveBackend) -> Cells:
    return screen.styled(screen.row - len(backend.region), screen.row)


@pytest.fixture
def replay_terminal(
    terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch
) -> FakeTerminal:
    """The fake terminal at `COLUMNS` by `ROWS`, in true color by default.

    The render thread never draws, so every frame comes from the test.
    """
    monkeypatch.setattr(Backend, "interval", 3600.0)
    monkeypatch.setenv("COLUMNS", str(COLUMNS))
    monkeypatch.setenv("LINES", str(ROWS))
    monkeypatch.setenv("COLORTERM", "truecolor")
    return terminal


DEPTHS: tuple[dict[str, str], ...] = (
    {"COLORTERM": "truecolor"},
    {},
    {"NO_COLOR": "1"},
)


def _replay(
    terminal: FakeTerminal, side: PanelPosition, width: Callable[[str], int]
) -> list[int]:
    """Replay random frames into a screen drawing with ``width``.

    Bars come and go in numbers beyond the rows, so a "+N more" row shows too.

    Returns:
        The frames whose live region differs from a full redraw.
    """
    chance = random.Random(side)
    options = Options(stream=terminal, panel_position=side)
    screen = _screen(width)
    mismatches = []
    with Bar(total=500, desc="outer", options=options) as outer:
        engine = engine_for(terminal)
        backend = engine.backend
        assert isinstance(backend, LiveBackend)
        inner: list[Bar] = []
        now = time.monotonic()
        for frame in range(150):
            with engine_lock():
                roll = chance.random()
                if len(inner) < ROWS + 2 and roll < 0.07:
                    inner.append(Bar(total=40, desc="inner", options=options))
                elif inner and roll < 0.1:
                    inner.pop(chance.randrange(len(inner))).close()
                elif roll < 0.14:
                    outer.desc = chance.choice(DESCRIPTIONS)
                elif roll < 0.18:
                    outer.status(
                        loss=chance.random() if chance.random() < 0.7 else None,
                        mood=chance.choice(("ok", "\u2500\u2500", "\U0001f321")),
                    )
                elif roll < 0.2:
                    outer.log(f"step {frame}")
                elif roll < 0.22:
                    outer.panel({"epoch": frame, "note": chance.choice(DESCRIPTIONS)})
                outer.update(chance.randrange(3))
                for bar in inner:
                    bar.update(chance.randrange(2))
                now += chance.choice((0.02, 1 / 12, 0.5))
                start = len(terminal.getvalue())
                engine.draw(now)
                screen.feed(terminal.getvalue()[start:])
                if _live_region(screen, backend) != _full_redraw(backend, width):
                    mismatches.append(frame)
        for bar in inner:
            bar.close()
    return mismatches


@pytest.mark.usefixtures("replay_terminal")
@pytest.mark.parametrize("width", WIDTHS.values(), ids=WIDTHS.keys())
def test_patched_frames_leave_what_full_redraws_would(
    monkeypatch: pytest.MonkeyPatch, width: Callable[[str], int]
) -> None:
    # Act: at each color depth with the panel on each side
    mismatches = {}
    sides: tuple[PanelPosition, ...] = ("below", "above")
    for depth, side in product(DEPTHS, sides):
        with monkeypatch.context() as environment:
            environment.delenv("COLORTERM")
            for name, value in depth.items():
                environment.setenv(name, value)
            mismatches[(*depth, side)] = _replay(FakeTerminal(), side, width)

    # Assert
    assert {case: frames for case, frames in mismatches.items() if frames} == {}


def test_a_running_bar_sends_a_fraction_of_a_full_redraw(
    replay_terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setenv("COLUMNS", "120")
    with Bar(
        total=10_000, desc="train", options=Options(stream=replay_terminal)
    ) as bar:
        engine = engine_for(replay_terminal)
        backend = engine.backend
        assert isinstance(backend, LiveBackend)
        now = time.monotonic()
        with engine_lock():
            bar.update(4000)
            engine.draw(now)

            # Act
            sent = []
            for _ in range(48):
                bar.update(10)
                now += 1 / 12
                start = len(replay_terminal.getvalue())
                engine.draw(now)
                sent.append(len(replay_terminal.getvalue()[start:].encode()))
            row = backend._line(backend.region[0].runs)

    # Assert: rewriting the row in full on every frame would send over 3x more
    full = len(f"{SYNC_BEGIN}\x1b[1F{row}\x1b[K\n\x1b[J{SYNC_END}".encode())
    assert sum(sent) / len(sent) < full / 3


@pytest.mark.usefixtures("replay_terminal")
def test_a_write_failing_as_a_session_ends_still_prints_its_last_lines() -> None:
    # Arrange
    terminal = FlakyTerminal()
    bar = Bar(total=3, desc="job", options=Options(stream=terminal, leave=True))
    bar.update(3)
    with engine_lock():
        engine_for(terminal).draw(time.monotonic())
    bar.write("written last")
    terminal.fail_next = True

    # Act
    bar.close()

    # Assert
    output = terminal.getvalue()
    assert "written last\n" in output
    assert output.index("written last") < output.rindex("done")
    assert output.endswith("\x1b[?25h")


def _patch(old: list[tuple[str, Ink]], new: list[tuple[str, Ink]]) -> str:
    """Return the writes patching a row drawn from runs ``old`` into ``new``."""
    backend = LiveBackend(FakeTerminal())
    return backend._patch(_drawn(new), _drawn(old))


def _patched_screen(
    old: list[tuple[str, Ink]],
    new: list[tuple[str, Ink]],
    width: Callable[[str], int] = char_width,
) -> tuple[Cells, Cells]:
    """Return the row patched from ``old`` to ``new``, and ``new`` drawn whole."""
    screen, fresh = _screen(width), _screen(width)
    screen.feed(WRAP_OFF + _patch([], old))
    screen.feed("\x1b[1F" + _patch(old, new))
    fresh.feed(WRAP_OFF + _patch([], new))
    return screen.styled(0, 1), fresh.styled(0, 1)


def test_runs_joining_the_cell_before_or_after_them_are_not_patched() -> None:
    # Arrange: terminals that join such a character to its neighbor's cell
    # would lose one when a patch draws either alone
    heads = "\ufe0f\u093e\U0001f3fb\U0001f3ff\u1161\u11ff\ud7b0\ud7fb"
    heads += "\U00016d63\U00016d6a\uff9e\uff9f\u0e33\u0eb3"
    tails = "\u1100\u115f\ua960\ua97f\u0d4e\U000111c2\U000111c3\U0001193f"
    tails += "\U00011941\U000113d1\U00011a3a\U00011a84\U00011a89\U00011d46"
    tails += "\U00011f02"

    # Act
    patchable = {
        f"U+{ord(char):04X}": row.patchable
        for char, row in [
            *(
                (head, _drawn([("\U0001f44d", PLAIN), (f"{head} ok", BOLD)]))
                for head in heads
            ),
            *(
                (tail, _drawn([(f"k{tail}", BOLD), ("\uac00 value", PLAIN)]))
                for tail in tails
            ),
        ]
    }

    # Assert
    assert [char for char, kept in patchable.items() if kept] == []


def test_patched_rows_match_rows_drawn_whole() -> None:
    # Arrange: drawn wide, box glyphs put each run after them a cell further
    # right than psnek measures
    gap = ("x" * 10, PLAIN)
    cases: list[tuple[list[Run], list[Run], Callable[[str], int]]] = [
        # a mark's run between changes would stack on the cell before it
        (
            [("1", BOLD), ("abcdefgh", PLAIN), ("\u0301z", BOLD), ("2", PLAIN)],
            [("3", BOLD), ("abcdefgh", PLAIN), ("\u0301z", BOLD), ("4", PLAIN)],
            char_width,
        ),
        (
            [("\u2840", PLAIN), ("z0\u2500", BOLD), ("xb", PLAIN)],
            [("\u2840", PLAIN), ("z0\u2500", BOLD)],
            char_width,
        ),
        (
            [("\u2500", PLAIN), gap, ("yy", BOLD)],
            [("\u2500", PLAIN), gap],
            _ambiguous_wide,
        ),
        (
            [("\u2500", PLAIN), gap, ("\u2500", PLAIN), gap, ("1", BOLD)],
            [("\u2500", PLAIN), gap, ("\u2500", PLAIN), gap, ("2", BOLD)],
            _ambiguous_wide,
        ),
    ]

    # Act
    rows = [_patched_screen(old, new, width) for old, new, width in cases]

    # Assert
    assert [
        index for index, (patched, whole) in enumerate(rows) if patched != whole
    ] == []


def test_patches_move_over_unchanged_runs_every_terminal_draws_alike() -> None:
    # Arrange
    tree = ("\u251c\u2500 ", PLAIN)
    label = ("x" * 20, PLAIN)
    cases = [
        ([("abc", BOLD)], [], "\x1b[K\n"),
        # the indent before tree glyphs is moved over, the glyphs drawn
        (
            [("      \u2514\u2500", PLAIN), label, ("1", BOLD)],
            [("      \u2514\u2500", PLAIN), label, ("2", BOLD)],
            "\x1b[6C\u2514\u2500\x1b[20C\x1b[1m2\x1b[0m\n",
        ),
        (
            [tree, label, ("1", BOLD)],
            [tree, label, ("2", BOLD)],
            "\u251c\u2500 \x1b[20C\x1b[1m2\x1b[0m\n",
        ),
        # spaces changing only their color look the same
        (
            [("ab", BOLD), ("   ", BOLD), ("cd", PLAIN)],
            [("ab", BOLD), ("   ", Ink(0x123456)), ("cd", PLAIN)],
            "\n",
        ),
        # four cells apart, rewriting the gap beats a move; 21 apart, it does not
        (
            [("1", PLAIN), (" of ", PLAIN), ("2", PLAIN), label],
            [("3", PLAIN), (" of ", PLAIN), ("4", PLAIN), label],
            "3 of 4\n",
        ),
        (
            [("1", PLAIN), (" " + "x" * 20, PLAIN), ("2", PLAIN)],
            [("3", PLAIN), (" " + "x" * 20, PLAIN), ("4", PLAIN)],
            "3\x1b[21C4\n",
        ),
        # changed runs of disputed width rewrite the row whole
        (
            [("\u2500" * 3, PLAIN), (" " + "x" * 20, PLAIN), ("1", PLAIN)],
            [("\u2500" * 4, PLAIN), ("x" * 20, PLAIN), ("1", PLAIN)],
            "\u2500" * 4 + "x" * 20 + "1\x1b[K\n",
        ),
    ]

    # Act
    written = [_patch(old, new) for old, new, _ in cases]

    # Assert
    assert written == [expected for _, _, expected in cases]
