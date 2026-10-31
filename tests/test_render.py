"""Tests for pure frame rendering, including golden gallery snapshots."""

import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import cast

import pytest

from psnek import _render, render
from psnek._frame import Frame, State
from psnek._glyphs import ASCII, BRAILLE, Glyphs, Style
from psnek._render import (
    GALLERY,
    Columns,
    clip,
    columns_for,
    gallery,
    layout,
    log_lines,
    more_bars,
    panel_runs,
    printed_runs,
    summary,
)
from psnek._style import (
    BOLD,
    DEFAULT_PALETTE,
    FAINT,
    PLAIN,
    ColorDepth,
    Ink,
    Palette,
    Run,
    to_plain,
)
from psnek._text import Scale, display_width

GOLDEN = Path(__file__).parent / "golden"
PLAIN_STYLE = Style(depth=ColorDepth.NONE)
FACELESS = replace(BRAILLE, comet_heads=None, stalled_heads=None, stuck_heads=None)
ASCII_STYLE = Style(glyphs=ASCII, depth=ColorDepth.NONE)


def test_braille_glyphs_are_single_width() -> None:
    for code in range(0x2800, 0x2900):
        assert unicodedata.east_asian_width(chr(code)) == "N"


def test_coil_fills_up_then_right_then_down() -> None:
    # Arrange: dots 7,3,2,1 (left column, bottom-up), then 4,5,6,8 (right, top-down)
    expected = ["⠀", "⡀", "⡄", "⡆", "⡇", "⡏", "⡟", "⡿", "⣿"]

    # Assert
    assert list(BRAILLE.partials) == expected
    assert BRAILLE.track == "⣀"


def test_ascii_steps_and_spinner_snake_are_distinct() -> None:
    assert len(set(ASCII.partials)) == len(ASCII.partials)
    assert ASCII.track not in ASCII.partials
    for frame in BRAILLE.spinner:
        assert bin(ord(frame) - 0x2800).count("1") == 3
    assert len(set(BRAILLE.spinner)) == 8


@pytest.mark.parametrize("glyphs", [BRAILLE, ASCII])
def test_layout_never_exceeds_width(glyphs: Glyphs) -> None:
    # Arrange
    style = Style(glyphs=glyphs, depth=ColorDepth.NONE)
    widths = (1, 2, 5, 12, 19, 20, 24, 33, 40, 57, 80, 120, 200)

    for (name, frame), width in product(GALLERY, widths):
        # Act
        line = to_plain(layout(frame, width, style, Columns(label=12))[0])

        # Assert
        assert display_width(line) <= width, (name, width)
        assert "\n" not in line
        if frame.state != "running" and width >= 40:
            assert frame.state in line, (name, width)


def test_layout_drops_fields_in_order_of_importance() -> None:
    # Arrange
    frame = Frame(
        n=5,
        total=10,
        elapsed=3,
        rate=2,
        desc="x",
        status=(("loss", "0.1"), ("lr", "3e-4")),
    )

    # Act
    lines = {
        width: to_plain(layout(frame, width, PLAIN_STYLE)[0])
        for width in (120, 65, 56, 44, 32, 20)
    }

    # Assert
    assert lines[120].endswith("2.0it/s  loss=0.1  lr=3e-4")
    assert lines[65].endswith("0:03<0:02    2.0it/s")
    assert lines[56].endswith("0:03<0:02")
    assert lines[44].endswith("5/10")
    assert lines[32].endswith("50.0%")
    assert "%" not in lines[20]


def test_status_is_reserved_only_while_the_bar_keeps_twenty_cells() -> None:
    # Arrange
    frame = Frame(n=5, total=10, elapsed=3, rate=2, desc="x", status=(("k", "v"),))

    # Act
    roomy, tight = (
        to_plain(layout(frame, width, PLAIN_STYLE)[0]) for width in (93, 92)
    )

    # Assert
    assert roomy.endswith("2.0it/s  k=v")
    assert roomy.count("\u28ff") + roomy.count("\u28c0") == 20
    assert tight.count("\u28ff") + tight.count("\u28c0") > 20


def test_a_status_wider_than_the_room_is_shortened_beside_a_twenty_cell_bar() -> None:
    # Arrange
    status = (("loss", "0.0243906250012345678"), ("lr", "3e-4"))
    frame = Frame(n=5, total=10, elapsed=3, rate=2, desc="x", status=status)

    # Act
    line = to_plain(layout(frame, 93, PLAIN_STYLE)[0])

    # Assert
    assert line.count("\u28ff") + line.count("\u28c0") == 20
    assert line.endswith("2.0it/s  loss=0.02439062500123...")
    assert display_width(line) == 93


def test_status_pairs_are_drawn_like_panel_records() -> None:
    # Arrange
    status = (("loss", "0.1"), ("val loss", "0.2"), ("a=b", "c"))
    frame = Frame(n=5, total=10, elapsed=3, rate=2, desc="x", status=status)
    tall = replace(frame, height=2)

    # Act
    one_row = layout(frame, 140, Style())[0]
    below = layout(tall, 140, Style())[1]

    # Assert
    for runs in (one_row, below):
        assert ("loss", BOLD) in runs
        assert ("val loss", BOLD) in runs
        assert ("a=b", BOLD) in runs
        assert runs.count(("=", FAINT)) == 3
        assert ("0.2", PLAIN) in runs


def test_a_shortened_status_pair_keeps_its_styling() -> None:
    # Act
    in_value = _render._cut_pair("note", "a very long status value", 12)
    in_key = _render._cut_pair("a" * 20, "1", 10)

    # Assert
    assert in_value == (("note", BOLD), ("=", FAINT), ("a ve...", PLAIN))
    assert in_key == (("aaaaaaa", BOLD), ("...", PLAIN))


def test_long_status_pairs_are_shortened_with_an_ellipsis() -> None:
    # Arrange
    frame = Frame(n=1, total=2, status=(("key", "a" * 60),))

    # Act
    line = to_plain(layout(frame, 100, PLAIN_STYLE)[0])

    # Assert
    assert display_width(line) <= 100
    assert "key=aaa" in line
    assert line.endswith("...")


def _rate_end(line: str) -> int:
    return max(line.find("/s") + len("/s"), line.find("/it") + len("/it"))


@pytest.mark.parametrize(
    "frames",
    [
        [
            Frame(n=count, total=1000, elapsed=elapsed, rate=rate, desc="x")
            for count, elapsed, rate in ((1, 1, 0.5), (10, 9, 12.3), (999, 70, 1234.0))
        ],
        [
            Frame(n=3, total=10, elapsed=5, rate=1.0),
            Frame(n=3, elapsed=5, rate=1.0),
            Frame(n=10, total=10, elapsed=5, rate=1.0, state="done"),
            Frame(n=3, total=10, elapsed=5, rate=1.0, state="failed"),
            Frame(n=3, total=10, elapsed=60, rate=1.0, state="stalled", idle=9),
        ],
    ],
)
def test_fields_keep_their_columns_across_values_and_states(
    frames: list[Frame],
) -> None:
    # Arrange
    columns = columns_for(frames, 33)

    # Act
    lines = [to_plain(layout(frame, 100, PLAIN_STYLE, columns)[0]) for frame in frames]

    # Assert: every field sits in a slot shared by the stack
    assert len({_rate_end(line) for line in lines}) == 1
    assert len({line.index("/", line.index("%")) for line in lines if "%" in line}) == 1


def test_columns_grow_from_the_floor_but_never_narrow_below_it() -> None:
    # Arrange
    floor = columns_for([Frame(n=1, status=(("loss", "0.02439"),))], 33)

    # Act
    narrower = columns_for([Frame(n=1, status=(("loss", "0.02"),))], 33, floor)
    wider = columns_for(
        [Frame(n=1, desc="train", status=(("lr", "3e-4" * 4),))], 33, floor
    )

    # Assert
    assert narrower == floor
    assert wider == replace(floor, label=5, status=len("lr=" + "3e-4" * 4))


def test_stacks_with_long_counts_and_durations_stay_aligned() -> None:
    # Arrange
    frames = [
        Frame(n=50, total=100, elapsed=5, rate=10, desc="a"),
        Frame(
            n=1.5 * 2**30,
            total=4 * 2**30,
            elapsed=264_225,
            rate=16.4 * 2**20,
            desc="b",
            unit="B",
            scale="iec",
            state="interrupted",
        ),
        Frame(
            n=3, total=10, elapsed=3700, rate=0.01, desc="c", state="stalled", idle=65
        ),
    ]
    columns = columns_for(frames, 33)

    # Act
    lines = [to_plain(layout(frame, 120, PLAIN_STYLE, columns)[0]) for frame in frames]

    # Assert: the rate column ends at the same cell in every row
    assert len({_rate_end(line) for line in lines}) == 1
    assert "3d01h" in lines[1]
    assert "interrupted" in lines[1]


def test_running_bars_never_claim_completion() -> None:
    # Act
    almost = to_plain(layout(Frame(n=9996, total=10000), 60, PLAIN_STYLE)[0])
    done = to_plain(layout(Frame(n=10000, total=10000), 60, PLAIN_STYLE)[0])

    # Assert
    assert "99.9%" in almost
    assert not almost.split()[1].endswith(BRAILLE.partials[8])
    assert "100.0%" in done


def test_tiny_widths_drop_the_percent_whole() -> None:
    # Act
    lines = [
        to_plain(layout(Frame(n=1000, total=1000, state="done"), width, PLAIN_STYLE)[0])
        for width in range(6, 30)
    ]

    # Assert
    assert all("100.0%" in line or "100" not in line for line in lines)
    assert any("100.0%" in line for line in lines)


def test_stuck_and_interrupted_are_named() -> None:
    # Act
    stuck = to_plain(
        layout(Frame(n=1, total=4, state="stuck", idle=95), 90, PLAIN_STYLE)[0]
    )
    stopped = summary(Frame(n=1, total=4, state="interrupted"))

    # Assert
    assert "stuck 1m35s" in stuck
    assert "interrupted" in stopped


def test_finished_unknown_total_draws_a_full_bar() -> None:
    # Act
    line = to_plain(layout(Frame(n=7, state="done"), 40, PLAIN_STYLE)[0])

    # Assert
    assert BRAILLE.partials[8] * 10 in line


def test_comet_head_leads_in_the_direction_of_travel() -> None:
    # Arrange
    forward = Frame(n=1, phase=0.6)
    backward = Frame(n=1, phase=2.2)

    def head_index(frame: Frame) -> int:
        runs = layout(frame, 40, Style())[0][2:]
        cells = [ink for text, ink in runs for _ in text]
        body = [i for i, ink in enumerate(cells) if ink.color is not None]
        base = DEFAULT_PALETTE.gradient[0]
        heads = [i for i in body if cells[i].color != base]
        return heads[0] - body[0]

    # Act / Assert
    assert head_index(forward) > 0
    assert head_index(backward) == 0


def test_ascii_comet_shows_its_direction() -> None:
    # Act
    forward = to_plain(layout(Frame(n=1, phase=0.6), 40, ASCII_STYLE)[0])
    backward = to_plain(layout(Frame(n=1, phase=2.2), 40, ASCII_STYLE)[0])

    # Assert
    assert "#>" in forward
    assert "<#" in backward


def test_labels_align_without_a_description() -> None:
    # Act
    columns = Columns(label=4)
    named = to_plain(
        layout(Frame(n=1, total=2, desc="abcd"), 60, PLAIN_STYLE, columns)[0]
    )
    blank = to_plain(layout(Frame(n=1, total=2), 60, PLAIN_STYLE, columns)[0])

    # Assert
    assert named.index("%") == blank.index("%")


def test_state_labels_use_the_terminal_foreground() -> None:
    # Act
    runs = layout(Frame(n=1, total=2, state="failed"), 60, Style())[0]

    # Assert
    assert ("failed", BOLD) in runs
    assert any(ink.color == DEFAULT_PALETTE.failed for _, ink in runs)


def test_metrics_draw_numbers_in_the_foreground_and_units_faint() -> None:
    # Act
    runs = layout(Frame(n=1, total=2, elapsed=3, rate=1.5, desc="d"), 100, Style())[0]

    # Assert
    texts = {text.strip(): ink for text, ink in runs}
    assert texts["d"] == PLAIN
    assert texts["0:03"] == texts["0:00"] == texts["1.5"] == PLAIN
    assert texts["<"] == texts["it/s"] == texts["/"] == FAINT
    assert texts["50.0%"] == BOLD


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "1.50GiB/4.00GiB",
            [("1.50", PLAIN), ("GiB/", FAINT), ("4.00", PLAIN), ("GiB", FAINT)],
        ),
        ("22.0kit/s", [("22.0", PLAIN), ("kit/s", FAINT)]),
        ("1:02:03<0:04", [("1:02:03", PLAIN), ("<", FAINT), ("0:04", PLAIN)]),
        ("--:--", [("--:--", FAINT)]),
        ("96.0k/96.0k", [("96.0k", PLAIN), ("/", FAINT), ("96.0k", PLAIN)]),
        ("1.50Ki/s", [("1.50Ki", PLAIN), ("/s", FAINT)]),
        ("33.3%", [("33.3%", PLAIN)]),
        ("  12it", [("  ", PLAIN), ("12", PLAIN), ("it", FAINT)]),
        ("-3s", [("-3", PLAIN), ("s", FAINT)]),
    ],
)
def test_quantity_keeps_bare_prefixes_and_percent_with_the_number(
    text: str, expected: list[tuple[str, Ink]]
) -> None:
    # Act
    runs = list(_render._quantity(text))

    # Assert
    assert runs == expected


def test_render_sanitizes_any_frame() -> None:
    # Act
    line = render(Frame(n=1, total=2, desc="\x1b[2Jx", unit="\x1b]0;t\x07B"), 80)

    # Assert
    assert "\x1b[2J" not in line
    assert "\x1b]0" not in line
    assert "\x07" not in line
    assert "\x1b" not in re.sub(r"\x1b\[[0-9;]*m", "", line)


def test_huge_descriptions_are_cut_to_a_bounded_prefix() -> None:
    # Arrange
    frame = Frame(n=1, total=2, desc="\x07" * 5_000_000)

    # Act
    line = to_plain(layout(frame, 80, PLAIN_STYLE)[0])

    # Assert
    assert "\\u0007" in line
    assert display_width(line) <= 80


def test_frames_reject_numbers_that_cannot_be_drawn() -> None:
    invalid: list[Callable[[], Frame]] = [
        lambda: Frame(n=float("nan")),
        lambda: Frame(elapsed=float("inf")),
        lambda: Frame(rate=float("nan")),
        lambda: Frame(total=-1),
        lambda: Frame(total=float("inf")),
        lambda: Frame(estimate=float("inf")),
    ]
    for build in invalid:
        with pytest.raises(ValueError, match="frame"):
            build()


def test_wide_description_is_truncated_not_wrapped() -> None:
    # Arrange
    frame = Frame(n=1, total=2, desc="日本語の説明" * 10)

    # Act
    line = to_plain(layout(frame, 50, PLAIN_STYLE)[0])

    # Assert
    assert display_width(line) <= 50
    assert "..." in line


def test_unknown_total_draws_a_moving_comet_without_a_percent_column() -> None:
    # Arrange
    frame = Frame(n=3, desc="stream")

    # Act
    early = to_plain(layout(frame, 90, PLAIN_STYLE)[0])
    later = to_plain(layout(replace(frame, phase=0.7), 90, PLAIN_STYLE)[0])
    known = to_plain(layout(replace(frame, total=9), 90, PLAIN_STYLE)[0])

    # Assert
    assert "%" not in early
    assert early != later
    assert len(early) == len(known)


def test_spinner_animates_with_phase() -> None:
    # Act
    glyphs = {
        layout(Frame(total=1, phase=step / 10), 40, Style())[0][0][0]
        for step in range(10)
    }

    # Assert
    assert len(glyphs) > 4


def test_summary_is_plain_and_complete() -> None:
    # Arrange
    frame = Frame(
        n=42, total=100, elapsed=12, rate=3.5, desc="job", status=(("loss", "0.1"),)
    )

    # Act
    line = summary(frame)

    # Assert
    assert line == "job: 42.0% 42/100 0:12<0:16 3.5it/s loss=0.1"
    assert summary(Frame(n=1, total=2, desc="a\x1b[31mb")).startswith("ab:")


def test_summary_marks_terminal_states() -> None:
    assert "0:00 done" in summary(Frame(n=1, total=1, state="done"))
    assert "failed" in summary(Frame(n=0, total=1, state="failed"))
    assert "stalled 12s" in summary(Frame(n=0, total=1, state="stalled", idle=12))
    assert "stuck 48s" in summary(Frame(n=0, total=1, state="stuck", idle=48))


def test_clip_and_panel_runs_fit() -> None:
    runs = panel_runs("x" * 200, 30, Style())
    assert display_width(to_plain(runs)) <= 30
    for width in (0, 1, 2):
        assert display_width(to_plain(panel_runs("hello", width, PLAIN_STYLE))) <= width
    assert to_plain(clip([("abc", PLAIN), ("def", PLAIN)], 4)) == "abcd"


def test_clip_measures_each_run_when_a_selector_follows_a_wide_one() -> None:
    # Arrange
    runs = [("\u65e5", PLAIN), ("\ufe0f", BOLD), ("ab", FAINT)]

    # Act
    clipped = clip(runs, 4)
    whole = clip(runs, 5)

    # Assert
    assert clipped == [*runs[:2], ("a", FAINT)]
    assert whole == runs


def test_styled_panel_lines_keep_their_ink_and_end_in_an_ellipsis() -> None:
    # Arrange
    line = (("key  ", BOLD), ("v" * 50, PLAIN))

    # Act
    fitting = panel_runs(line, 80, PLAIN_STYLE)
    cut = panel_runs(line, 30, PLAIN_STYLE)

    # Assert
    assert fitting[1:] == list(line)
    assert cut[1] == ("key  ", BOLD)
    assert to_plain(cut).endswith("v...")
    assert display_width(to_plain(cut)) == 30


@pytest.mark.parametrize(
    ("name", "render_gallery"),
    [
        ("braille.txt", lambda: gallery(100, PLAIN_STYLE)),
        ("narrow.txt", lambda: gallery(56, PLAIN_STYLE)),
        ("ascii.txt", lambda: gallery(100, ASCII_STYLE)),
        ("wide.txt", lambda: gallery(140, ASCII_STYLE)),
        ("ansi.txt", lambda: gallery(100, Style(depth=ColorDepth.EIGHT_BIT))),
        ("html.html", lambda: gallery(100, html=True)),
        ("tall.txt", lambda: gallery(100, PLAIN_STYLE, height=2)),
        ("tower.txt", lambda: gallery(120, PLAIN_STYLE, phase=0.35, height=4)),
        ("tall_ascii.txt", lambda: gallery(80, ASCII_STYLE, height=3)),
    ],
)
def test_gallery_matches_golden(
    name: str, render_gallery: Callable[[], str], update_golden: bool
) -> None:
    # Arrange
    path = GOLDEN / f"gallery_{name}"

    # Act
    rendered = render_gallery() + "\n"

    # Assert
    if update_golden:
        path.write_text(rendered, encoding="utf-8")
    assert rendered == path.read_text(encoding="utf-8")


def test_gallery_legend_is_colored() -> None:
    # Act
    legend = gallery(80).split("\n")[0]

    # Assert
    assert legend.count("\x1b[38;2;") == len(BRAILLE.partials) - 1


def test_gallery_html_is_markup_without_escape_sequences() -> None:
    # Act
    markup = gallery(80, html=True)
    plain = gallery(80, Style(depth=ColorDepth.NONE), html=True)

    # Assert
    assert markup.startswith("<pre")
    assert "\x1b" not in markup
    assert "color:" in markup
    assert "color:" not in plain


def test_a_palette_without_a_distinct_head_color_makes_the_head_bold() -> None:
    # Arrange
    palette = replace(DEFAULT_PALETTE, gradient=(0x1A87A1,), shimmer=0x1A87A1)

    for depth in (ColorDepth.EIGHT_BIT, ColorDepth.TRUE_COLOR):
        # Act
        style = Style(glyphs=FACELESS, palette=palette, depth=depth)
        runs = layout(Frame(n=1, phase=0.6), 40, style)[0]
        inks = {ink for _, ink in runs[2:] if ink.color is not None}

        # Assert
        assert Ink(0x1A87A1, bold=True) in inks


def test_an_eta_beyond_floats_reads_as_the_longest_duration() -> None:
    # Arrange
    frame = Frame(n=0, total=1e308, elapsed=1, rate=1e-9)

    # Act
    line = to_plain(layout(frame, 80, PLAIN_STYLE)[0])

    # Assert
    assert "0:01<99d+" in line


def test_an_estimated_eta_overrides_the_rate_and_is_marked_when_uncertain() -> None:
    # Arrange
    base = Frame(n=1, total=4, elapsed=10, rate=0.1)

    # Act
    lines = [
        summary(replace(base, estimate=25.0)),
        summary(replace(base, estimate=25.0, approximate=True)),
        summary(replace(base, rate=None, approximate=True)),
        summary(replace(base, state="done", approximate=True)),
    ]

    # Assert
    assert [line.split()[2] for line in lines] == [
        "0:10<0:25",
        "0:10<~0:25",
        "0:10<--:--",
        "0:10",
    ]


def test_only_status_pairs_that_can_be_shown_are_formatted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    frame = Frame(n=1, status=tuple((f"m{i}", str(i)) for i in range(10_000)))
    formatted: list[str] = []
    visible = _render._visible

    def counting(text: str) -> str:
        formatted.append(text)
        return visible(text)

    monkeypatch.setattr(_render, "_visible", counting)

    # Act
    layout(frame, 80, PLAIN_STYLE, columns_for([frame], 40))[0]

    # Assert
    assert len(formatted) < 100


def test_comet_wears_a_smiley_face_on_its_leading_end() -> None:
    # Arrange
    assert BRAILLE.comet_heads is not None
    face = BRAILLE.comet_heads[0]

    # Act
    forward = to_plain(layout(Frame(n=1, phase=0.6), 40, PLAIN_STYLE)[0])
    backward = to_plain(layout(Frame(n=1, phase=2.2), 40, PLAIN_STYLE)[0])

    # Assert
    assert forward.count(face) == backward.count(face) == 1
    assert forward.split(face)[1].startswith(BRAILLE.track)
    assert not backward.split(face)[1].startswith(BRAILLE.track)


def test_comet_face_shows_the_mood_of_each_health_state() -> None:
    # Arrange
    moods: dict[State, tuple[str, str] | None] = {
        "running": BRAILLE.comet_heads,
        "stalled": BRAILLE.stalled_heads,
        "stuck": BRAILLE.stuck_heads,
    }

    # Act
    lines = {
        state: to_plain(layout(Frame(n=1, state=state, phase=0.6), 40, PLAIN_STYLE)[0])
        for state in moods
    }

    # Assert
    faces = {state: heads[0] for state, heads in moods.items() if heads is not None}
    assert len(set(faces.values())) == len(moods)
    for state, line in lines.items():
        assert [face for face in faces.values() if face in line] == [faces[state]]


def test_comet_face_is_never_bold() -> None:
    # Act
    runs = layout(Frame(n=1, state="stuck", phase=0.6), 40, Style())[0]

    # Assert
    assert BRAILLE.stuck_heads is not None
    assert (BRAILLE.stuck_heads[0], Ink(DEFAULT_PALETTE.failed)) in runs


def test_comet_drops_its_face_without_room_to_travel_its_width() -> None:
    # Arrange
    assert BRAILLE.comet_heads is not None
    face = BRAILLE.comet_heads[0]
    frames = [Frame(n=1, phase=step / 10) for step in range(30)]

    # Act
    narrow, roomy = (
        [
            to_plain(_render._comet_cells(frame, cells, PLAIN_STYLE)[0])
            for frame in frames
        ]
        for cells in (2 * len(face) - 1, 2 * len(face))
    )

    # Assert
    assert all(face not in row for row in narrow)
    assert all(face in row for row in roomy)
    assert len(set(narrow)) > 1


def test_faceless_comet_is_marked_without_color() -> None:
    # Arrange
    style = replace(PLAIN_STYLE, glyphs=FACELESS)

    # Act
    line = to_plain(layout(Frame(n=1, phase=0.6), 40, style)[0])

    # Assert
    assert BRAILLE.partials[8] in line


def test_unknown_totals_that_end_badly_show_an_empty_colored_track() -> None:
    # Act
    runs = layout(Frame(n=4, state="failed", phase=1.9), 40, Style())[0]

    # Assert
    track = [ink for text, ink in runs if set(text) == {BRAILLE.track}]
    assert track == [Ink(DEFAULT_PALETTE.failed)]


def test_hidden_counts_name_their_noun_in_singular_or_plural() -> None:
    assert to_plain(more_bars(1, 40)) == "  +1 more bar"
    assert to_plain(more_bars(2, 40, "earlier", "line")) == "  +2 earlier lines"


def test_frames_reject_malformed_text_status_and_choices() -> None:
    for status in ({"lr": "3e-4"}, (("loss", 0.1),), (("lonely",),), [("a", "b")]):
        with pytest.raises(TypeError, match="status"):
            Frame(status=cast("tuple[tuple[str, str], ...]", status))
    builds: list[tuple[Callable[[], Frame], type[Exception]]] = [
        (lambda: Frame(desc=cast("str", 1)), TypeError),
        (lambda: Frame(unit=cast("str", None)), TypeError),
        (lambda: Frame(scale=cast("Scale", "metric")), ValueError),
        (lambda: Frame(state=cast("State", "sleeping")), ValueError),
        (lambda: Frame(branches=cast("tuple[bool, ...]", [True])), TypeError),
        (lambda: Frame(branches=cast("tuple[bool, ...]", (1,))), TypeError),
        (lambda: Frame(approximate=cast("bool", 1)), TypeError),
    ]
    for build, error in builds:
        with pytest.raises(error):
            build()


def test_styles_that_cannot_draw_are_rejected() -> None:
    invalid: list[Callable[[], object]] = [
        lambda: replace(ASCII, spinner=()),
        lambda: replace(ASCII, partials=("#",)),
        lambda: replace(BRAILLE, dot_bits=(1,)),
        lambda: replace(ASCII, comet_heads=(">>", "<")),
        lambda: replace(ASCII, comet_heads=("", "")),
        lambda: replace(ASCII, stuck_heads=("<>", "<>")),
        lambda: replace(ASCII, stalled_heads=("", "")),
        lambda: replace(ASCII, comet_heads=None, stuck_heads=("x", "x")),
        lambda: Palette(gradient=()),
        lambda: Palette(gradient=cast("tuple[int, ...]", [0x00877F])),
        lambda: Palette(stalled=-1),
        lambda: Palette(stalled=cast("int", 1.5)),
        lambda: replace(BRAILLE, dot_bits=(-1,) * 8),
        lambda: replace(BRAILLE, dot_bits=(1,) * 8),
        lambda: replace(BRAILLE, tree=cast("tuple[str, str, str, str]", ("| ",))),
        lambda: replace(BRAILLE, tree=("|", "  ", "|-", "`-")),
        lambda: Style(depth=cast("ColorDepth", 16)),
    ]
    for build in invalid:
        with pytest.raises(ValueError):
            build()
    with pytest.raises(TypeError, match="Glyphs and a Palette"):
        Style(glyphs=cast("Glyphs", None))
    with pytest.raises(TypeError, match="Glyphs and a Palette"):
        Style(palette=cast("Palette", "x"))
    assert Style(depth=cast("ColorDepth", 8)).depth is ColorDepth.EIGHT_BIT


@pytest.mark.parametrize(
    ("glyphs", "prefixes"),
    [
        (BRAILLE, ["", "\u251c\u2500", "\u2502 \u2514\u2500", "\u2514\u2500"]),
        (ASCII, ["", "|-", "| `-", "`-"]),
    ],
)
def test_nested_bars_branch_from_their_parent_and_stay_aligned(
    glyphs: Glyphs, prefixes: list[str]
) -> None:
    # Arrange: a top bar, a child with a sibling below, its child, the sibling
    tree: list[tuple[bool, ...]] = [(), (True,), (True, False), (False,)]
    frames = [
        Frame(n=1, total=4, desc=desc, branches=branches)
        for desc, branches in zip(("top", "a", "a1", "b"), tree, strict=True)
    ]
    style = Style(glyphs=glyphs, depth=ColorDepth.NONE)
    columns = columns_for(frames, 30)

    # Act
    lines = [to_plain(layout(frame, 90, style, columns)[0]) for frame in frames]

    # Assert: the tree precedes each spinner, and every bar starts in one column
    spinner = 1
    for line, prefix, frame in zip(lines, prefixes, frames, strict=True):
        assert line.startswith(prefix)
        assert line[len(prefix) + spinner :].lstrip().startswith(frame.desc + " ")
    assert columns.label == len("| `-") + len("a1")
    assert len({line.index("25.0%") for line in lines}) == 1


def test_tree_glyphs_are_faint() -> None:
    # Act
    runs = layout(Frame(desc="child", branches=(False,)), 60, Style())[0]

    # Assert
    assert runs[0] == ("\u2514\u2500", FAINT)


def test_render_rejects_a_width_below_one() -> None:
    with pytest.raises(ValueError, match="width"):
        render(Frame(), 0)


def test_wide_units_keep_the_rate_column_aligned() -> None:
    # Arrange
    frames = [
        Frame(n=1, total=9, rate=12.0, unit="画像"),
        Frame(n=1, total=9, rate=1.5, unit="画像"),
    ]
    columns = columns_for(frames, 20)

    # Act
    lines = [to_plain(layout(frame, 100, PLAIN_STYLE, columns)[0]) for frame in frames]

    # Assert
    assert len({display_width(line[: line.index("/s")]) for line in lines}) == 1


def test_a_stack_mixing_known_and_unknown_totals_keeps_the_percent_column() -> None:
    # Arrange
    frames = [Frame(n=3, desc="stream"), Frame(n=3, total=9, desc="known")]
    columns = columns_for(frames, 20)

    # Act
    lines = [to_plain(layout(frame, 70, PLAIN_STYLE, columns)[0]) for frame in frames]

    # Assert
    assert columns.percent == 6
    assert columns_for(frames[:1], 20).percent == 0
    assert lines[0].index("0:00<") == lines[1].index("0:00<")


def test_a_zero_total_is_full_once_anything_was_counted() -> None:
    assert Frame(n=0, total=0).fraction == 0.0
    assert Frame(n=5, total=0).fraction == 1.0
    assert Frame(n=0, total=0, state="done").fraction == 1.0


def test_huge_animation_phases_still_render() -> None:
    for frame in (
        Frame(phase=1e308),
        Frame(phase=1e308, state="stalled"),
        Frame(n=1, total=2, phase=1e308, state="stuck"),
    ):
        assert 0 < display_width(render(frame, 40, PLAIN_STYLE)) <= 40


def test_the_state_label_is_kept_while_it_fits_without_a_bar() -> None:
    # Arrange
    frame = Frame(n=5, total=10, desc="x", elapsed=3, rate=2, state="stalled")

    # Act
    lines = {
        width: to_plain(layout(frame, width, ASCII_STYLE)[0]) for width in (19, 20)
    }

    # Assert
    assert "stalled" not in lines[19]
    assert lines[20] == "| x   stalled 0s"


def test_stacked_bars_align_when_the_label_cap_is_at_its_minimum() -> None:
    # Arrange
    frames = [Frame(n=5, total=10, desc="a" * 8), Frame(n=5, total=10, desc="bb")]
    columns = columns_for(frames, 11 // 3)

    # Act
    long, short = (
        to_plain(layout(frame, 11, ASCII_STYLE, columns)[0]) for frame in frames
    )

    # Assert
    assert long == "| a... ##__"
    assert short == "| bb   ##__"


def test_a_running_bar_lines_up_with_a_stalled_bar_at_every_width() -> None:
    # Arrange
    running = Frame(n=37, total=100, elapsed=60, rate=2.0)
    stalled = replace(running, state="stalled", idle=9)

    # Act
    rows = {
        width: [
            to_plain(
                layout(
                    frame,
                    width,
                    ASCII_STYLE,
                    columns_for([running, stalled], width // 3),
                )[0]
            )
            for frame in (running, stalled)
        ]
        for width in range(12, 101)
    }

    # Assert
    for width, (running_line, stalled_line) in rows.items():
        assert stalled_line.startswith(running_line), width
        assert "stalled" in stalled_line or width < 18, width


def test_a_stalled_bar_keeps_its_width_as_the_idle_time_grows() -> None:
    # Arrange
    stalled = Frame(n=5, total=10, desc="x", elapsed=3, rate=2, state="stalled")
    ages = (9, 59, 60, 3599, 3600)

    # Act
    frames = [replace(stalled, idle=age) for age in ages]
    lines = [to_plain(layout(frame, 60, ASCII_STYLE)[0]) for frame in frames]

    # Assert
    assert len({line.index("  ") for line in lines}) == 1


def test_blank_glyphs_render_at_tiny_widths() -> None:
    # Arrange
    blank = replace(ASCII, spinner=(" ",), track=" ", done=" ")
    style = Style(glyphs=blank, depth=ColorDepth.NONE)

    # Act
    line = render(Frame(), 2, style)

    # Assert
    assert line == ""


WARN = Ink(DEFAULT_PALETTE.stalled, bold=True)
FAILED = Ink(DEFAULT_PALETTE.failed, bold=True)


def test_log_lines_raise_salience_with_the_level() -> None:
    # Arrange: a level between the standard ones takes the one below
    levels = {
        logging.DEBUG: (("debug", FAINT), FAINT),
        logging.INFO: (("info ", FAINT), PLAIN),
        logging.WARNING: (("warn ", WARN), PLAIN),
        logging.ERROR: (("error", FAILED), PLAIN),
        logging.CRITICAL: (("fatal", FAILED), BOLD),
        5: (("debug", FAINT), FAINT),
        25: (("info ", FAINT), PLAIN),
        99: (("fatal", FAILED), BOLD),
    }

    # Act
    tagged = {level: log_lines(["message", "more"], level) for level in levels}

    # Assert
    assert tagged == {
        level: [(tag, (" message", ink)), (("      ", PLAIN), ("more", ink))]
        for level, (tag, ink) in levels.items()
    }


def test_log_lines_put_a_faint_time_first_and_end_without_blanks() -> None:
    # Act
    first, second, third, blank, last = log_lines(
        ["two\x1b[2J lines  ", "more\t", "\tend \u2800", "", "last"],
        logging.ERROR,
        "09:30:00",
    )

    # Assert: continuations indent under the text, and an empty line prints
    # nothing, in the panel only its gutter
    assert first == (("09:30:00 ", FAINT), ("error", FAILED), (" two lines", PLAIN))
    assert list(map(to_plain, (second, third, last))) == [
        " " * 15 + "more",
        " " * 15 + "        end",
        " " * 15 + "last",
    ]
    assert to_plain(printed_runs(blank, BRAILLE)) == ""
    assert to_plain(panel_runs(blank, 20, Style(glyphs=BRAILLE))) == (
        BRAILLE.gutter.rstrip()
    )


def test_a_cause_beside_its_state_is_cut_before_the_state_or_bar_gives_way() -> None:
    # Arrange
    cause = "UnicodeDecodeError"
    frames = {
        height: Frame(3, 10, 2, 1.5, "decode", state="failed", height=height)
        for height in (1, 3)
    }
    short = Frame(12, 20, 0.2, 75.4, "index shards", state="failed", cause="ValueError")
    wide = replace(frames[1], cause="\u65e5\u672c\u8a9e")

    def drawn(frame: Frame, width: int) -> str:
        return "\n".join(map(to_plain, layout(frame, width, ASCII_STYLE)))

    # Act
    shown = {
        (height, width): (drawn(replace(bare, cause=cause), width), drawn(bare, width))
        for height, bare in frames.items()
        for width in range(12, 80)
    }

    # Assert: wherever the state shows alone, it shows with the cause, cut
    # short rather than squeezing the bar below its least width, and a cut
    # cause keeps two cells at least, as one wide character
    for case, (with_cause, alone) in shown.items():
        assert ("failed" in with_cause) == ("failed" in alone), case
        assert with_cause.count("#") >= min(alone.count("#"), 3), case
        after = with_cause.partition("failed")[2].split("\n")[0]
        assert after in ("", f" {cause}") or (
            after.endswith("...") and len(after) >= len(" Un...")
        ), case
    assert cause in shown[1, 79][0]
    assert all(
        drawn(short, width).endswith("  failed ValueError") for width in (59, 99)
    )
    assert drawn(wide, 33).endswith("failed \u65e5...")


def test_a_long_cause_narrows_every_bar_in_a_stack_alike(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    widths: list[int] = []
    bar_cells = _render._bar_cells

    def measured(frame: Frame, cells: int, style: Style) -> list[list[Run]]:
        widths.append(cells)
        return bar_cells(frame, cells, style)

    monkeypatch.setattr(_render, "_bar_cells", measured)

    def stack(width: int, frames: list[Frame]) -> tuple[list[int], list[str]]:
        columns = columns_for(frames, max(width // 3, 4))
        widths.clear()
        rows = [
            "\n".join(map(to_plain, layout(f, width, ASCII_STYLE, columns)))
            for f in frames
        ]
        return list(widths), rows

    def stacks(height: int, cause: str) -> list[Frame]:
        return [
            Frame(3, 10, 2, 1.5, "decode", state="failed", cause=cause, height=height),
            Frame(5, 10, 125, 1.5, "other", state="stalled", idle=125, height=height),
            Frame(10, 10, 2, 1.5, "third", state="done", height=height),
        ]

    # Act
    shown = {
        (height, width): (
            stack(width, stacks(height, "UnicodeDecodeError")),
            stack(width, stacks(height, "")),
        )
        for height in (1, 3)
        for width in range(16, 100)
    }
    stalled, running = stacks(1, "")[1], replace(stacks(1, "")[1], state="running")
    carried = columns_for([running], 20, columns_for([stalled], 20))

    # Assert: the bars line up wherever they do without the cause (a taller
    # bar's state ends a row at its own width), the cause takes no label
    # from the others, and the stack's floor holds between draws
    for case, ((bars, rows), (bare_bars, bare_rows)) in shown.items():
        assert len(bars) == len(bare_bars), case
        assert len(set(bars)) <= 1 or len(set(bare_bars)) > 1, case
        labels = ("stalled", "done")
        for row, bare_row, label in zip(rows[1:], bare_rows[1:], labels, strict=True):
            assert (label in row) == (label in bare_row), case
    (_, (failed_row, _, _)), _ = shown[1, 28]
    assert failed_row.endswith("failed Unic...")
    assert carried.bare == columns_for([stalled], 20).bare
    assert carried.bare > columns_for([running], 20).bare


def test_a_cause_shows_only_on_stopped_bars_and_in_summaries() -> None:
    # Arrange
    stopped = Frame(3, 10, 7, 0.4, "cancel", state="interrupted", cause="Ctrl-C\x1b[2J")
    others = [replace(stopped, state=state) for state in ("running", "done")]

    # Act
    shown = summary(stopped)
    hidden = [summary(frame) for frame in others]

    # Assert
    assert shown == "cancel: 30.0% 3/10 0:07 interrupted Ctrl-C 2.50s/it"
    assert all("Ctrl-C" not in line for line in hidden)
    with pytest.raises(TypeError, match="cause"):
        Frame(cause=cast("str", 1))


def test_bars_without_a_gradient_draw_in_the_default_color() -> None:
    # Arrange
    frames = [
        Frame(n=55, total=100, gradient=False),
        Frame(n=100, total=100, state="done", gradient=False),
        Frame(n=4, gradient=False),
        Frame(n=4, state="done", gradient=False),
    ]
    styles = [Style(), Style(glyphs=FACELESS), Style(depth=ColorDepth.EIGHT_BIT)]

    # Act
    colors = {
        ink.color
        for frame, style, height, step in product(frames, styles, (1, 2), range(40))
        for row in layout(replace(frame, height=height, phase=step / 13), 60, style)
        for _, ink in row
    }

    # Assert
    assert colors == {None}


def test_bars_without_a_gradient_keep_their_state_colors() -> None:
    # Arrange
    states: list[tuple[State, int]] = [
        ("stalled", DEFAULT_PALETTE.stalled),
        ("stuck", DEFAULT_PALETTE.failed),
        ("failed", DEFAULT_PALETTE.failed),
        ("interrupted", DEFAULT_PALETTE.interrupted),
    ]

    for state, color in states:
        # Act
        runs = layout(Frame(n=40, total=100, state=state, gradient=False), 60, Style())

        # Assert
        assert color in {ink.color for _, ink in runs[0]}


def test_bars_with_their_own_stops_draw_with_them() -> None:
    # Arrange
    stops = (0xFF0000, 0x0000FF)
    style = Style(glyphs=FACELESS)

    # Act
    running = layout(Frame(n=55, total=100, gradient=stops, phase=50.0), 60, style)
    done = layout(Frame(n=10, total=10, state="done", gradient=stops), 60, style)
    comet = layout(Frame(n=4, phase=50.0, gradient=stops), 60, style)

    # Assert
    _, *bar = (ink.color for _, ink in running[0] if ink.color is not None)
    assert bar[0] == stops[0]
    assert done[0][0] == (FACELESS.done, Ink(stops[-1], bold=True))
    palette = set(DEFAULT_PALETTE.gradient)
    assert all(
        ink.color not in palette for row in (running, done, comet) for _, ink in row[0]
    )
    assert any(ink.color is not None for _, ink in comet[0])


@pytest.mark.parametrize("gradient", [1, (), (0x1000000,), (True,), [0xFF0000]])
def test_frames_reject_a_gradient_that_is_not_a_bool_or_rgb_stops(
    gradient: object,
) -> None:
    with pytest.raises(TypeError, match="gradient"):
        Frame(gradient=cast("bool", gradient))
