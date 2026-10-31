"""Performance regression tests for the traffic of a running notebook output.

A notebook output is resent whole on every update, so these tests pin how its
bytes and drawing work scale with the lines printed into it and the sessions
run in its cell. Where a regression would only make updates larger, the rate
cap is lifted, as delaying large updates would otherwise hide it.
"""

import io
import logging
import re
import time
from collections.abc import Iterable

import pytest

from psnek import Bar, Options, _backends, _engine
from psnek._backends import (
    NOTEBOOK_KEPT,
    NOTEBOOK_LIVE_BARS,
    NOTEBOOK_LIVE_LINES,
)
from psnek._frame import Frame
from psnek._glyphs import Style
from psnek._render import Columns, columns_for
from tests.fakes import (
    FakeEvents,
    FakeHandle,
    engine_for,
    engine_lock,
    settle_idle_engines,
    shown_text,
)

LINE = "processed record 00000 from shard 07"
DRAW_EVERY = 10
LIVE_UPDATE_BUDGET = 4_500
_REPEATED_STYLE = re.compile(r'<span style="([^"]*)">[^<]*</span> *<span style="\1">')


def _new_cell(fake_ipython: list[FakeHandle]) -> None:
    """Forget the cell's display, so the next bar starts afresh."""
    fake_ipython.clear()
    _engine._OUTPUTS.clear()


def _print_lines(
    fake_ipython: list[FakeHandle], lines: int, level: int = logging.INFO
) -> list[dict[str, str]]:
    """Log ``lines`` lines into a running bar, drawing every `DRAW_EVERY` lines.

    The engine lock is held throughout, so only these draws send updates.

    Returns:
        The updates sent while the bar ran.
    """
    stream = io.StringIO()
    with (
        engine_lock(),
        Bar(total=1, desc="ingest", options=Options(stream=stream)) as bar,
    ):
        engine = engine_for(stream)
        for line in range(1, lines + 1):
            bar.log(LINE, level=level)
            if line % DRAW_EVERY == 0:
                engine.draw(time.monotonic())
        live = list(fake_ipython[0].updates)
        bar.update()
    settle_idle_engines()
    return live


def _bytes(update: dict[str, str]) -> int:
    return sum(len(data.encode()) for data in update.values())


@pytest.mark.usefixtures("uncapped")
def test_live_updates_stay_small_as_lines_are_printed_and_the_final_keeps_all(
    fake_ipython: list[FakeHandle],
) -> None:
    # Arrange
    half = _print_lines(fake_ipython, 1000, logging.WARNING)
    _new_cell(fake_ipython)

    # Act
    full = _print_lines(fake_ipython, 2000, logging.WARNING)

    # Assert: only the digits of the earlier-lines count differ between the
    # last live updates, resending every line would make the total grow
    # fourfold, and the final update has every line
    assert abs(_bytes(full[-1]) - _bytes(half[-1])) <= len("2000") - len("980")
    assert sum(map(_bytes, full)) / sum(map(_bytes, half)) < 2.1
    assert "text/plain" not in full[-1]
    assert full[-1]["text/html"].count("white-space:pre-wrap") == 1
    assert _bytes(full[-1]) <= LIVE_UPDATE_BUDGET
    final = fake_ipython[0].updates[-1]
    assert final["text/plain"].count(LINE) == 2000
    assert final["text/html"].count(LINE) == 2000


def test_no_span_repeats_the_style_of_the_one_before(
    fake_ipython: list[FakeHandle],
) -> None:
    # Arrange
    stream = io.StringIO()

    # Act
    with Bar(total=10, desc="train", options=Options(stream=stream)) as bar:
        bar.update(4)
        bar.status(loss=0.137, lr=0.001)
        bar.log("checkpoint saved", level=logging.WARNING)
        with engine_lock():
            engine_for(stream).draw(time.monotonic())
        live = fake_ipython[0].updates[-1]["text/html"]
        bar.update(6)
    settle_idle_engines()

    # Assert
    final = fake_ipython[0].updates[-1]["text/html"]
    assert _REPEATED_STYLE.search(live) is None
    assert _REPEATED_STYLE.search(final) is None


@pytest.mark.usefixtures("uncapped", "fake_ipython")
def test_a_live_draw_lays_out_only_the_lines_it_shows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    laid_out: list[int] = []
    html = vars(_backends._Text)["html"]

    def counted(text: _backends._Text) -> str:
        laid_out.append(len(text.rows))
        rendered: str = html.fget(text)
        return rendered

    monkeypatch.setattr(_backends._Text, "html", property(counted))
    stream = io.StringIO()
    with (
        engine_lock(),
        Bar(total=1, desc="ingest", options=Options(stream=stream)) as bar,
    ):
        engine = engine_for(stream)
        for _ in range(5000):
            bar.write(LINE)

        # Act
        for step in range(30):
            bar.status(step=step)
            engine.draw(time.monotonic())
        live = list(laid_out)
        bar.update()
    settle_idle_engines()

    # Assert: each of the 30 updates lays out the shown lines, not all 5000
    assert len(live) == 30
    assert max(live) == NOTEBOOK_LIVE_LINES


@pytest.mark.usefixtures("uncapped", "never_busy")
def test_traffic_grows_linearly_with_the_sessions_in_a_cell(
    fake_ipython: list[FakeHandle], cell_events: FakeEvents
) -> None:
    # Arrange
    def traffic(sessions: int) -> int:
        stream = io.StringIO()
        with engine_lock():
            for session in range(sessions):
                options = Options(stream=stream)
                with Bar(total=1, desc=f"epoch {session}", options=options) as bar:
                    for _ in range(10):
                        bar.write(LINE)
                    bar.update()
            cell_events.end_cell()
        settle_idle_engines()
        total = sum(map(_bytes, fake_ipython[0].updates))
        _new_cell(fake_ipython)
        cell_events.start_cell()
        return total

    # Act
    few = traffic(50)
    many = traffic(200)

    # Assert: resending the cell's history at each session's end would make
    # four times the sessions send about sixteen times the bytes
    assert many / few < 4.5


def test_a_large_session_end_does_not_delay_the_next_session(
    fake_ipython: list[FakeHandle],
) -> None:
    # Arrange
    stream = io.StringIO()
    with engine_lock():
        with Bar(total=1, desc="load", options=Options(stream=stream)) as load:
            for _ in range(5000):
                load.write(LINE)
            load.update()
        settle_idle_engines()
        sent = len(fake_ipython[0].updates)

        # Act
        with Bar(total=10, desc="train", options=Options(stream=stream)) as train:
            for _ in range(3):
                train.update()
                engine_for(stream).draw(time.monotonic())
            live = fake_ipython[0].updates[sent:]
            train.update(7)

    # Assert: the complete update was large enough to delay a live one
    assert _bytes(fake_ipython[0].updates[sent - 1]) / _backends.NOTEBOOK_RATE > 1
    assert len(live) == 3


def test_a_large_live_update_delays_the_next_to_hold_the_rate(
    fake_ipython: list[FakeHandle], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(_backends, "NOTEBOOK_RATE", 100)
    stream = io.StringIO()
    with (
        engine_lock(),
        Bar(total=10, desc="ingest", options=Options(stream=stream)) as bar,
    ):
        engine = engine_for(stream)
        engine.draw(time.monotonic())
        output = _engine._OUTPUTS[next(iter(_engine._OUTPUTS))]
        sent = len(fake_ipython[0].updates)

        # Act
        bar.update(3)
        engine.draw(time.monotonic())
        held = len(fake_ipython[0].updates)
        output.sent_at -= output.wait
        engine.draw(time.monotonic())
        resumed = len(fake_ipython[0].updates)
        wait, latest = output.wait, fake_ipython[0].updates[-1]
        bar.update(7)
    settle_idle_engines()

    # Assert: the wait is the update's UTF-8 size over the rate, its Braille
    # glyphs counting three bytes each, and the final update is sent at once
    characters = sum(map(len, latest.values()))
    assert wait == _bytes(latest) / 100
    assert _bytes(latest) > characters
    assert held == sent
    assert resumed == sent + 1
    assert "done" in fake_ipython[0].updates[-1]["text/plain"]


def test_live_draws_skip_plain_text_and_measure_only_running_bars(
    fake_ipython: list[FakeHandle], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    texts: list[Frame] = []
    measured: list[int] = []
    bar_text = _backends._bar_text
    narrowest = columns_for((), 0)

    def counted_text(frame: Frame, columns: Columns, style: Style) -> str:
        texts.append(frame)
        return bar_text(frame, columns, style)

    def counted_columns(
        frames: Iterable[Frame], label_cap: int, floor: Columns = narrowest
    ) -> Columns:
        frames = list(frames)
        measured.append(len(frames))
        return columns_for(frames, label_cap, floor)

    monkeypatch.setattr(_backends, "_bar_text", counted_text)
    monkeypatch.setattr(_backends, "columns_for", counted_columns)
    stream = io.StringIO()
    options = Options(stream=stream, leave=True)
    with engine_lock(), Bar(total=10, desc="epochs", options=options) as outer:
        for epoch in range(12):
            with Bar(total=1, desc=f"epoch {epoch}", options=options) as done:
                done.update()
        engine = engine_for(stream)
        texts.clear()
        measured.clear()

        # Act
        for _ in range(5):
            outer.update()
            engine.draw(time.monotonic())
        live_texts, live_measured = len(texts), max(measured)
        outer.update(5)
    settle_idle_engines()

    # Assert: each kept bar was measured when kept, not on every draw, and
    # plain text is laid out only for the final update
    assert live_texts == 0
    assert live_measured == 1
    assert texts
    assert fake_ipython[0].updates[-1]["text/plain"].count("epoch ") == 12


def _kept_bars(fake_ipython: list[FakeHandle], count: int) -> dict[str, str]:
    """Keep ``count`` finished bars above a running one, then draw it live.

    Returns:
        The live update.
    """
    stream = io.StringIO()
    options = Options(stream=stream, leave=True)
    with engine_lock(), Bar(total=2, desc="files", options=options) as outer:
        for file in range(count):
            with Bar(total=1, desc=f"file {file:02d}", options=options) as done:
                done.update()
        outer.update()
        engine_for(stream).draw(time.monotonic())
        live = fake_ipython[0].updates[-1]
        outer.update()
    settle_idle_engines()
    return live


def test_a_running_output_shows_only_the_most_recent_kept_bars(
    fake_ipython: list[FakeHandle],
) -> None:
    # Arrange
    few = _kept_bars(fake_ipython, 12)
    few_final = fake_ipython[0].updates[-1]["text/plain"]
    _new_cell(fake_ipython)

    # Act
    many = _kept_bars(fake_ipython, NOTEBOOK_KEPT + 5)
    many_final = fake_ipython[0].updates[-1]["text/plain"]

    # Assert: live, the five newest kept bars show under a counter for the
    # rest, in the same markup however many are kept; the final output keeps
    # every bar up to `NOTEBOOK_KEPT`, the outer one too, counting the rest
    live = shown_text(few)
    assert live.splitlines()[0] == "  +7 earlier bars"
    assert [f"file {file:02d}" in live for file in range(12)] == [False] * 7 + [
        True
    ] * NOTEBOOK_LIVE_BARS
    assert "earlier" not in few_final
    assert all(f"file {file:02d}" in few_final for file in range(12))
    assert shown_text(many).splitlines()[0] == f"  +{NOTEBOOK_KEPT} earlier bars"
    assert abs(_bytes(many) - _bytes(few)) < 100
    assert many_final.splitlines()[0] == "  +6 earlier bars"
    assert many_final.count("file ") == NOTEBOOK_KEPT - 1
