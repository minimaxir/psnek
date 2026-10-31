"""Tests for bars that span several rows, from rendering to live and notebook output."""

import io
import os
import re
import time
from dataclasses import replace
from itertools import chain, pairwise, product

import pytest

from psnek import Bar, Options, render, track
from psnek._frame import Frame, State
from psnek._glyphs import _FACES, ASCII, BRAILLE, Glyphs, Style, _face
from psnek._render import GALLERY, Columns, _bar_cells, _path, columns_for, layout
from psnek._style import ColorDepth, Ink, to_plain
from psnek._text import display_width
from tests.fakes import (
    FakeHandle,
    FakeTerminal,
    engine_for,
    engine_lock,
    screen_text,
    settle_idle_engines,
)
from tests.vt import Screen

PLAIN_STYLE = Style(depth=ColorDepth.NONE)
ASCII_STYLE = Style(glyphs=ASCII, depth=ColorDepth.NONE)
FACELESS = replace(BRAILLE, comet_heads=None, stalled_heads=None, stuck_heads=None)
HEIGHTS = (2, 3, 4)
TRACK = "⣀"
BLANK = "\u2800"
LONG_DESC = "fine-tune llama-8b on support tickets epoch 2/3 (warmup done)"
PAIRS = (
    ("loss", "0.1372"),
    ("acc", "0.913"),
    ("lr", "3.0e-4"),
    ("grad_norm", "1.27"),
    ("tokens/s", "48.2k"),
    ("gpu", "97%"),
)


def _rows(frame: Frame, width: int, style: Style = PLAIN_STYLE) -> list[str]:
    return [to_plain(row) for row in layout(frame, width, style)]


def _rows_in(frame: Frame, width: int, columns: Columns) -> list[str]:
    return [to_plain(row) for row in layout(frame, width, PLAIN_STYLE, columns)]


def _dots(glyph: str) -> set[int]:
    """Return the raised dots of a Braille ``glyph`` (none for a space)."""
    if glyph == " ":
        return set()
    bits = ord(glyph) - 0x2800
    return {dot for dot in range(1, 9) if bits & 1 << (dot - 1)}


def _count_end(row: str) -> int:
    match = re.search(r"\d(?=\s+\d+:\d\d<)", row)
    assert match is not None
    return match.end()


def test_the_fill_path_climbs_the_left_dots_then_descends_the_right() -> None:
    # Act
    one, two = _path(1), _path(2)

    # Assert
    assert one == tuple((0, step) for step in range(8))
    assert two[:8] == (
        *((1, step) for step in range(4)),
        *((0, step) for step in range(4)),
    )
    assert two[8:] == (
        *((0, step) for step in range(4, 8)),
        *((1, step) for step in range(4, 8)),
    )


@pytest.mark.parametrize("height", HEIGHTS)
def test_each_fill_step_raises_one_more_dot_in_a_tall_column(height: int) -> None:
    # Arrange
    steps = 8 * height
    frames = [Frame(n=step, total=steps, height=height) for step in range(1, steps)]

    # Act
    columns = [
        [
            " " if run[1].faint else run[0]
            for row in _bar_cells(frame, 1, PLAIN_STYLE)
            for run in row
        ]
        for frame in frames
    ]

    # Assert
    raised = [
        {(row, dot) for row, glyph in enumerate(column) for dot in _dots(glyph)}
        for column in columns
    ]
    every = {(row, dot) for row in range(height) for dot in range(1, 9)}
    assert [len(dots) for dots in raised] == list(range(1, steps))
    assert all(before < after for before, after in pairwise(raised))
    assert raised[0] == {(height - 1, 7)}
    assert raised[-1] == every - {(height - 1, 8)}


def test_one_row_faces_keep_their_kaomoji() -> None:
    # Act
    faces = [_face(draw, 1)[0] for draw in _FACES.values()]

    # Assert
    assert faces == ["⢌⣀⡡", "⠛⣤⠛", "⠕⣀⠪"]
    assert list(_FACES) == faces


@pytest.mark.parametrize("height", (1, *HEIGHTS, 8))
def test_faces_grow_with_the_bar_and_rest_on_its_baseline(height: int) -> None:
    # Act
    faces = [_face(draw, height) for draw in _FACES.values()]

    # Assert
    assert len(set(faces)) == len(faces)
    for rows in faces:
        dots = [[len(_dots(glyph)) for glyph in row] for row in rows]
        assert len(rows) == height
        assert {display_width(row) for row in rows} == {2 * height + 1}
        assert set().union(*map(_dots, rows[0])) & {1, 4}
        assert set().union(*map(_dots, rows[-1])) & {7, 8}
        assert dots == [row[::-1] for row in dots]


@pytest.mark.parametrize(
    ("state", "height"), list(product(("running", "stalled", "stuck"), HEIGHTS))
)
def test_a_tall_comet_wears_its_mood_as_tall_as_the_bar(
    state: State, height: int
) -> None:
    # Arrange
    heads = {
        "running": BRAILLE.comet_heads,
        "stalled": BRAILLE.stalled_heads,
        "stuck": BRAILLE.stuck_heads,
    }[state]
    assert heads is not None
    face = _face(_FACES[heads[0]], height)

    # Act
    rows = _rows(Frame(n=1, state=state, phase=0.6, height=height), 100)

    # Assert
    shown = [(row, part.rstrip("\u2800")) for row, part in zip(rows, face, strict=True)]
    assert len({row.index(part) for row, part in shown if part}) == 1


@pytest.mark.parametrize("height", HEIGHTS)
def test_a_tall_comet_too_narrow_for_its_face_still_slithers(height: int) -> None:
    # Arrange
    face = _face(_FACES[BRAILLE.comet_heads[0]], height)  # type: ignore[index]
    cells = 2 * display_width(face[0]) - 1
    frames = [Frame(n=1, phase=step / 10, height=height) for step in range(30)]

    # Act
    drawn = [
        tuple(map(to_plain, _bar_cells(frame, cells, Style()))) for frame in frames
    ]
    roomy = [to_plain(row) for row in _bar_cells(frames[6], cells + 1, Style())]

    # Assert
    visible = [part for part in face if part.strip("\u2800")]
    assert not any(part in "".join(rows) for rows in drawn for part in visible)
    assert len(set(drawn)) > 1
    assert all(
        part.strip("\u2800") in row for part, row in zip(face, roomy, strict=True)
    )


def test_an_ascii_comet_head_leads_every_row() -> None:
    # Act
    forward = _rows(Frame(n=1, phase=0.6, height=3), 40, ASCII_STYLE)
    backward = _rows(Frame(n=1, phase=2.0, height=3), 40, ASCII_STYLE)

    # Assert
    assert len({row.index(">") for row in forward}) == 1
    assert len({row.index("<") for row in backward}) == 1


def test_a_faceless_tall_comet_marks_only_its_raised_lead_cells() -> None:
    # Arrange
    frames = [Frame(n=1, phase=phase / 10, height=2) for phase in range(30)]
    shaded, plain = (
        Style(glyphs=FACELESS, depth=depth)
        for depth in (ColorDepth.TRUE_COLOR, ColorDepth.NONE)
    )

    # Act
    drawn = [
        (_bar_cells(frame, 20, shaded), _bar_cells(frame, 20, plain))
        for frame in frames
    ]

    # Assert
    for colored, uncolored in drawn:
        for colored_row, uncolored_row in zip(colored, uncolored, strict=True):
            raised = [run[0] not in (" ", TRACK) for run in colored_row]
            marked = [run[0] not in (" ", TRACK) for run in uncolored_row]
            assert raised == marked


@pytest.mark.parametrize(
    ("glyphs", "height"), list(product((BRAILLE, ASCII, FACELESS), HEIGHTS))
)
def test_tall_layouts_span_their_height_and_never_exceed_the_width(
    glyphs: Glyphs, height: int
) -> None:
    # Arrange
    style = Style(glyphs=glyphs, depth=ColorDepth.NONE)
    frames = [
        *(replace(frame, height=height) for _, frame in GALLERY),
        Frame(5, 9, 3, 1.5, LONG_DESC, status=PAIRS, height=height, phase=0.3),
        Frame(5, None, 3, 1.5, LONG_DESC, status=PAIRS, height=height, phase=2.1),
    ]

    # Act
    layouts = [
        (width, [to_plain(row) for row in layout(frame, width, style)])
        for frame, width in product(frames, range(1, 141, 7))
    ]

    # Assert
    for width, rows in layouts:
        assert len(rows) == height
        assert all(display_width(row) <= width for row in rows)


def test_every_row_of_a_tall_bar_draws_the_same_number_of_braille_cells() -> None:
    # Arrange
    frames = [
        Frame(37, 90, 3, 12.1, "docs", status=PAIRS, height=4, phase=0.4),
        Frame(37, None, 3, 12.1, "peps", status=PAIRS, height=4, phase=0.8),
        Frame(3, 8, 1, 2.0, "shard", height=3, branches=(True, False)),
    ]

    # Act
    layouts = [_rows(frame, 100) for frame in frames]

    # Assert
    for rows in layouts:
        bar_end = rows[0].rindex("⣿") + 1 if "⣿" in rows[0] else 30
        braille = {
            sum(0x2800 <= ord(char) <= 0x28FF for char in row[:bar_end]) for row in rows
        }
        assert len(braille) == 1, rows


def test_only_the_bottom_row_holds_the_track() -> None:
    # Act
    empty = _rows(Frame(n=0, total=10, height=3), 60)
    failed = layout(Frame(n=4, state="failed", height=3), 40, Style())

    # Assert
    assert TRACK * 10 in empty[2]
    assert BLANK * 10 in empty[0]
    assert not empty[1]
    assert TRACK not in empty[0] + empty[1]
    failed_bar = _bar_cells(Frame(n=4, state="failed", height=3), 10, Style())
    assert [to_plain(row) for row in failed_bar[:2]] == [BLANK * 10] * 2
    assert all(TRACK not in to_plain(row) for row in failed[:2])
    track_run = next(run for run in failed[2] if TRACK in run[0])
    assert track_run[1] == Ink(Style().palette.failed)


def test_a_tall_bar_wraps_its_description_instead_of_cutting_it() -> None:
    # Arrange
    frame = Frame(n=1, total=4, desc=LONG_DESC)

    # Act
    one = _rows(frame, 100)
    two = _rows(replace(frame, height=2), 100)

    # Assert
    assert "..." in one[0]
    assert " ".join(row[2:34] for row in two).split() == LONG_DESC.split()


def test_a_tall_bar_lays_its_status_in_columns_below_its_metrics() -> None:
    # Arrange
    frame = Frame(5, 9, 3, 1.5, "train", status=PAIRS, height=4)
    shown = [f"{key}={value}" for key, value in PAIRS]

    # Act
    rows = _rows(frame, 100)

    # Assert
    pairs = [[part for part in row.split() if "=" in part] for row in rows]
    assert "1.5it/s" in rows[0]
    assert not pairs[0]
    assert list(chain(*zip(*pairs[1:], strict=True))) == shown
    for column in (shown[:3], shown[3:]):
        lined_up = zip(rows[1:], column, strict=True)
        assert len({row.index(pair) for row, pair in lined_up}) == 1


def test_a_tall_bar_keeps_its_metrics_and_state_on_the_first_row() -> None:
    # Arrange
    frame = Frame(3, 3, 13, 0.231, LONG_DESC, status=PAIRS, state="done", height=4)

    # Act
    rows = _rows(frame, 100)

    # Assert
    assert rows[0].split()[-5:] == ["100.0%", "3/3", "0:13", "4.33s/it", "done"]
    assert all("=" in row for row in rows[1:])


def test_a_tall_bar_ends_its_first_row_with_its_state_in_a_wide_stack() -> None:
    # Arrange
    crawl = Frame(3, 3, 12.4, 0.242, LONG_DESC, status=PAIRS, height=4)
    page = Frame(30, None, 7.2, 4.1, "peps", state="stalled", idle=7, branches=(False,))
    columns = columns_for([crawl, page], 33)
    running, done = crawl, replace(crawl, state="done")

    # Act
    rows = {frame.state: _rows_in(frame, 100, columns) for frame in (running, done)}

    # Assert
    assert rows["done"][0].split()[-1] == "done"
    assert all("=" in row for row in rows["done"][1:])
    assert len({row[0].index("100.0%") for row in rows.values()}) == 1


@pytest.mark.parametrize(
    ("n", "total"), [(31, 360), (251, 360), (360, 360), (512, None)]
)
def test_a_tall_bar_starts_its_status_under_its_first_value(
    n: int, total: int | None
) -> None:
    # Arrange
    frame = Frame(n, total, 3.1, 81.5, LONG_DESC, status=PAIRS, height=2, phase=0.3)

    # Act
    top, below = _rows(frame, 110)

    # Assert
    first = top.split()[-4 if total else -3]
    assert top.index(first) == below.index("loss=")


def test_status_columns_that_do_not_fit_are_dropped() -> None:
    # Arrange
    frame = Frame(5, 9, 3, 1.5, "train", status=PAIRS, height=2)

    # Act
    rows = _rows(frame, 60)

    # Assert
    assert "loss=0.1372" in rows[1]
    assert "gpu=97%" not in "".join(rows)
    assert all(display_width(row) <= 60 for row in rows)


def test_a_tall_bar_wraps_its_metrics_rather_than_dropping_them() -> None:
    # Arrange
    frame = Frame(5, 9, 3, 1.5, "train", status=PAIRS)

    # Act
    one = " ".join(_rows(frame, 50))
    two = " ".join(_rows(replace(frame, height=2), 50))

    # Assert
    for metric in ("55.6%", "5/9", "0:03<0:02", "1.5it/s"):
        assert metric in two
    assert "1.5it/s" not in one


def test_a_status_pair_wider_than_any_row_is_shortened_on_its_own_row() -> None:
    # Arrange
    frame = Frame(5, 9, desc="x", status=(("a", "1"), ("note", "v" * 200)), height=3)

    # Act
    rows = _rows(frame, 80)

    # Assert
    assert rows[1].endswith("a=1")
    assert rows[2].endswith("...")
    assert "note=v" in rows[2]


def test_stacked_tall_bars_keep_their_columns_aligned() -> None:
    # Arrange
    frames = [
        Frame(5, 9, 3, 1.5, "short", height=2),
        Frame(812, 1200, 41, 19.8, LONG_DESC, state="stalled", idle=9, height=2),
        Frame(40, None, 30, 4.8, "stream", height=2, phase=0.4),
    ]
    columns = columns_for(frames, 33)

    # Act
    tops = [to_plain(layout(frame, 100, PLAIN_STYLE, columns)[0]) for frame in frames]

    # Assert
    assert len({_count_end(top) for top in tops}) == 1
    assert len({top.index("/s") for top in tops}) == 1
    assert tops[0].index("%") == tops[1].index("%")


def test_tall_nested_bars_draw_guides_down_to_their_children() -> None:
    # Arrange
    frames = [
        Frame(1, 3, desc="epochs", height=2, children=True),
        Frame(10, 40, desc="batches", height=2, branches=(True,), children=True),
        Frame(3, 8, desc="shard", height=2, branches=(True, False)),
        Frame(2, 9, desc="eval", height=2, branches=(False,)),
    ]
    columns = columns_for(frames, 33)
    spin = BRAILLE.spinner[0]

    # Act
    prefixes = [
        to_plain(row)[:4]
        for frame in frames
        for row in layout(frame, 100, PLAIN_STYLE, columns)
    ]

    # Assert
    assert prefixes == [
        f"{spin} ep",
        "│   ",
        f"├─{spin} ",
        "│ │ ",
        "│ └─",
        "│   ",
        f"└─{spin} ",
        "  \u2800 ",
    ]


def test_a_one_row_bar_ignores_whether_it_has_children() -> None:
    # Arrange
    frame = Frame(1, 3, desc="epochs", phase=0.4)

    # Act
    alone = layout(frame, 80, Style())
    parent = layout(replace(frame, children=True), 80, Style())

    # Assert
    assert alone == parent


def test_render_draws_one_line_per_row() -> None:
    # Act
    text = render(Frame(n=1, total=2, desc="d", height=3), 60)

    # Assert
    assert len(text.split("\n")) == 3


def test_render_can_draw_a_tall_bar_as_notebook_html() -> None:
    # Arrange
    frame = Frame(n=512, elapsed=75, rate=6.8, desc="crawl", phase=0.8, height=2)

    # Act
    page = render(frame, 90, html=True)

    # Assert
    assert page.startswith('<pre style="')
    assert page.endswith("</span></pre>")
    assert page.count("<br>") == 1
    assert "\x1b" not in page
    assert '<span style="opacity:.6">it/s</span>' in page


def test_frames_reject_heights_and_children_that_cannot_be_drawn() -> None:
    with pytest.raises(ValueError, match="height"):
        Frame(height=0)
    with pytest.raises(TypeError, match="height"):
        Frame(height=True)
    with pytest.raises(TypeError, match="children"):
        Frame(children=1)  # type: ignore[arg-type]


def test_a_live_tall_bar_leaves_all_its_rows(terminal: FakeTerminal) -> None:
    # Act
    options = Options(stream=terminal, height=3)
    for _ in track(range(3), LONG_DESC, options=options):
        pass

    # Assert
    lines = screen_text(terminal)
    assert len(lines) == 3
    assert "100.0%" in lines[0]
    assert "(warmup done)" in "".join(lines)


def test_the_live_region_fits_tall_bars_to_the_terminal_height(
    terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setenv("LINES", "6")
    options = Options(stream=terminal, height=2)

    # Act
    bars = [Bar(total=1, desc=f"bar {index}", options=options) for index in range(5)]
    with engine_lock():
        engine_for(terminal).draw(time.monotonic())
    during = screen_text(terminal)
    for bar in reversed(bars):
        bar.close()

    # Assert
    assert len(during) == 5
    assert during[-1] == "  +3 more bars"


def test_a_live_tall_parent_links_its_rows_to_its_children(
    terminal: FakeTerminal,
) -> None:
    # Arrange
    options = Options(stream=terminal, height=2)

    # Act
    with Bar(total=1, desc="outer", options=options) as outer:
        with Bar(total=1, desc="inner", options=options) as inner:
            with engine_lock():
                engine_for(terminal).draw(time.monotonic())
            during = screen_text(terminal)
            inner.update()
        outer.update()

    # Assert
    assert len(during) == 4
    assert [line[:2] for line in during[1:]] == ["│ ", "└─", "  "]


def test_live_tall_output_passes_the_strict_terminal_model(
    terminal: FakeTerminal,
) -> None:
    # Act
    options = Options(stream=terminal, height=4)
    with Bar(desc=LONG_DESC, options=options) as bar:
        for step in range(5):
            bar.status(step=step, **dict(PAIRS))
            bar.update()
            with engine_lock():
                engine_for(terminal).draw(time.monotonic() + step)

    # Assert
    screen = Screen(int(os.environ["COLUMNS"]), int(os.environ["LINES"]))
    screen.feed(terminal.getvalue())
    assert len(screen.text()) == 4


def test_a_notebook_shows_every_row_of_a_tall_bar(
    fake_ipython: list[FakeHandle],
) -> None:
    # Arrange
    stream = io.StringIO()

    # Act
    for _ in track(range(3), "nb", options=Options(stream=stream, height=2)):
        pass
    settle_idle_engines()

    # Assert
    final = fake_ipython[0].updates[-1]
    assert len(final["text/plain"].splitlines()) == 2
    assert final["text/html"].count("<br>") == 1
