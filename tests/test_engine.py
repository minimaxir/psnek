"""Tests for the live, log, and notebook engines and output capture."""

import contextlib
import io
import logging
import os
import re
import signal
import sys
import threading
import time
import types
from collections.abc import Iterator
from typing import TextIO, cast

import pytest

import psnek
from psnek import Bar, Options, _backends, _bar, _deferred, _engine, _text, track
from psnek._backends import NOTEBOOK_LIVE_LINES
from psnek._frame import Frame
from psnek._glyphs import ASCII, BRAILLE
from psnek._style import (
    DEFAULT_PALETTE,
    FAINT,
    ColorDepth,
    Ink,
    PanelLine,
    to_ansi,
    to_plain,
)
from tests.fakes import (
    QUIET,
    CountingStdout,
    FailingStream,
    FakeEvents,
    FakeHandle,
    FakeTerminal,
    FlakyTerminal,
    engine_for,
    engine_lock,
    screen_text,
    settle_idle_engines,
    shown_text,
    take_steps,
)


def _plain(lines: list[PanelLine]) -> list[str]:
    return [line if isinstance(line, str) else to_plain(line) for line in lines]


@pytest.fixture
def logger() -> Iterator[logging.Logger]:
    """A non-propagating logger whose handlers are removed afterwards."""
    instance = logging.getLogger("psnek.test.engine")
    instance.propagate = False
    yield instance
    instance.handlers.clear()
    instance.propagate = True


def test_live_bar_leaves_only_its_final_line(terminal: FakeTerminal) -> None:
    # Act
    for _ in track(range(3), "work", options=Options(stream=terminal)):
        pass

    # Assert
    lines = screen_text(terminal)
    assert len(lines) == 1
    assert "work" in lines[0]
    assert "100.0%" in lines[0]
    assert "3/3" in lines[0]
    assert terminal.getvalue().endswith("\x1b[?7h\x1b[?25h")


def test_live_redraw_is_synchronized_and_hides_cursor(terminal: FakeTerminal) -> None:
    # Act
    with Bar(total=2, options=Options(stream=terminal)) as bar:
        bar.update()

    # Assert
    output = terminal.getvalue()
    assert output.startswith("\x1b[?2026h\x1b[1G\x1b[?25l\x1b[?7l")
    assert output.count("\x1b[?2026h") == output.count("\x1b[?2026l")


def test_unchanged_lines_are_not_rewritten(terminal: FakeTerminal) -> None:
    # Arrange
    options = Options(stream=terminal)
    with Bar(total=4, desc="top", options=options) as top, engine_lock():
        child = Bar(total=4, desc="child", options=options)
        engine = engine_for(terminal)
        now = time.monotonic()
        engine.draw(now)
        before = len(terminal.getvalue())

        # Act
        child.update()
        engine.draw(now)
        redraw = terminal.getvalue()[before:]
        child.update(3)
        child.close()
        top.update(4)

    # Assert: the top line is skipped with a bare newline, and only the
    # child's changed cells are rewritten, after moves to their columns
    assert redraw.startswith("\x1b[?2026h\x1b[2F\n\x1b[")
    assert "25.0%" in redraw
    assert "top" not in redraw
    assert "child" not in redraw


def test_bars_started_while_another_is_joining_share_one_engine(
    terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    options = Options(stream=terminal)
    flush = _engine.flush_stdout
    racing: list[Bar] = []
    started = threading.Event()

    def flush_while_another_bar_starts(stream: TextIO) -> None:
        if not started.is_set():
            started.set()
            racing.append(Bar(desc="second", options=options))
        flush(stream)

    monkeypatch.setattr(_engine, "flush_stdout", flush_while_another_bar_starts)

    # Act
    first = Bar(desc="first", options=options)
    engines = [
        engine for engine in _engine.ENGINES.values() if engine.stream is terminal
    ]
    joined = len(engines[0].sources)
    first.close()
    racing[0].close()

    # Assert
    assert len(engines) == 1
    assert joined == 2


def test_redraws_track_growth_logs_and_shrinking(terminal: FakeTerminal) -> None:
    # Arrange
    options = Options(stream=terminal, panel_lines=2)
    outer = Bar(total=4, desc="outer", options=options)
    engine = engine_for(terminal)

    # Act: each step is one deterministic frame
    with engine_lock():
        inner = Bar(total=2, desc="inner", options=options)
        outer.log("first")
        outer.log("second")
        engine.draw(time.monotonic())
        outer.log("third")
        engine.draw(time.monotonic())
        during = screen_text(terminal)
        inner.update(2)
        inner.close()
        outer.update(4)
        outer.close()

    # Assert
    assert during[0] == "  info  first"
    assert "outer" in during[1]
    assert "inner" in during[2]
    assert during[3:] == ["⡇ info  second", "⡇ info  third"]
    final = screen_text(terminal)
    assert final[0] == "  info  first"
    assert "outer" in final[1]
    assert "100.0%" in final[1]
    assert final[2:] == ["  info  second", "  info  third"]


def test_printing_is_captured_into_panel_and_kept(terminal: FakeTerminal) -> None:
    # Arrange
    options = Options(stream=terminal, panel_lines=2)

    # Act
    for item in track(range(4), "job", options=options):
        print(f"item {item}")
    restored = sys.stdout

    # Assert
    lines = screen_text(terminal)
    assert lines[:2] == ["item 0", "item 1"]
    assert "job" in lines[2]
    assert lines[3:] == ["item 2", "item 3"]
    assert not isinstance(restored, _engine.Capture)


def test_unterminated_output_is_previewed_then_kept(terminal: FakeTerminal) -> None:
    # Act
    with Bar(total=1, desc="job", options=Options(stream=terminal)) as bar:
        print("loading 10%\rloading 20%", end="")
        preview = engine_for(terminal).panel_view()
        bar.update()

    # Assert
    assert preview == ["loading 20%"]
    assert screen_text(terminal)[-1] == "loading 20%"


def test_log_lines_are_cleaned(terminal: FakeTerminal) -> None:
    # Arrange
    cases = {"crlf text\r": "crlf text", "old text\rnew": "new text", "a\x1b[2Jb": "ab"}

    # Act
    with Bar(total=1, options=Options(stream=terminal)) as bar:
        for raw in cases:
            bar.log(raw)
            bar.write(raw)
        bar.update()

    # Assert
    screen = screen_text(terminal)
    assert all(screen.count(shown) == 1 for shown in cases.values())
    assert all(screen.count(f"  info  {shown}") == 1 for shown in cases.values())


def _tree_view(terminal: FakeTerminal) -> list[str]:
    """Draw now, then return each screen line's tree prefix and description."""
    with engine_lock():
        engine_for(terminal).draw(time.monotonic())
        lines = screen_text(terminal)
    return [re.sub(r"[\u2800-\u28ff] (\S+).*", r"\1", line) for line in lines if line]


def test_nested_loops_hang_below_their_parent(terminal: FakeTerminal) -> None:
    # Arrange
    options = Options(stream=terminal)

    # Act
    with Bar(total=1, desc="outer", options=options) as outer:
        with Bar(total=1, desc="middle", options=options) as middle:
            with Bar(total=1, desc="inner", options=options) as inner:
                nested = _tree_view(terminal)
                inner.update()
            middle.update()
        after = _tree_view(terminal)
        outer.update()

    # Assert
    assert nested == ["outer", "\u2514\u2500middle", "  \u2514\u2500inner"]
    assert after == ["outer"]


def test_bars_from_other_threads_are_siblings_in_tree_order(
    terminal: FakeTerminal,
) -> None:
    # Arrange
    options = Options(stream=terminal)
    bars: list[Bar] = []
    resume = threading.Event()

    def first_worker() -> None:
        bars.append(Bar(total=1, desc="a", options=options))
        resume.wait()
        bars.append(Bar(total=1, desc="a1", options=options))

    # Act: a1 starts after b, from the thread that started a
    with Bar(total=1, desc="top", options=options) as top:
        worker = threading.Thread(target=first_worker)
        worker.start()
        while not bars:
            time.sleep(0.001)
        second = threading.Thread(
            target=lambda: bars.append(Bar(total=1, desc="b", options=options))
        )
        second.start()
        second.join()
        resume.set()
        worker.join()
        tree = _tree_view(terminal)
        for bar in bars:
            bar.close()
        top.update()

    # Assert
    assert tree == [
        "top",
        "\u251c\u2500a",
        "\u2502 \u2514\u2500a1",
        "\u2514\u2500b",
    ]


def test_children_of_a_finished_bar_move_up_to_its_parent(
    terminal: FakeTerminal,
) -> None:
    # Arrange
    options = Options(stream=terminal)
    with Bar(total=1, desc="outer", options=options) as outer:
        middle = Bar(total=1, desc="middle", options=options)
        inner = Bar(total=1, desc="inner", options=options)

        # Act
        middle.close()
        tree = _tree_view(terminal)
        inner.close()
        outer.update()

    # Assert
    assert tree == ["outer", "\u2514\u2500inner"]


@pytest.mark.parametrize("clear", [False, True])
def test_panel_note_stays_after_the_bars_unless_cleared(
    terminal: FakeTerminal, clear: bool
) -> None:
    # Arrange
    options = Options(stream=terminal, panel_clear=clear)
    with Bar(total=1, desc="job", options=options) as bar:
        bar.log("first log")

        # Act
        bar.panel("status: busy\nworker 2 idle")
        replaced = engine_for(terminal).panel_view()
        bar.update()

    # Assert: logged lines go above, and the note stays below the final bar
    assert replaced == ["status: busy", "worker 2 idle"]
    lines = [line for line in screen_text(terminal) if line]
    assert lines[0] == "  info  first log"
    assert "job" in lines[1]
    gutter = BRAILLE.gutter
    kept = [f"{gutter}status: busy", f"{gutter}worker 2 idle"]
    assert lines[2:] == ([] if clear else kept)


def test_kept_panel_note_above_the_bars_stays_above_them(
    terminal: FakeTerminal,
) -> None:
    # Arrange
    options = Options(stream=terminal, panel_position="above")

    # Act
    with Bar(total=1, desc="job", options=options) as bar:
        bar.panel({"shard": 3})
        bar.log("logged")
        bar.update()

    # Assert
    lines = [line for line in screen_text(terminal) if line]
    assert lines[:2] == [f"{BRAILLE.gutter}shard  3", "  info  logged"]
    assert "job" in lines[2]


def test_oversized_panel_text_is_elided(terminal: FakeTerminal) -> None:
    # Act
    with Bar(total=1, options=Options(stream=terminal, panel_lines=3)) as bar:
        bar.panel("\n".join(f"line {i}" for i in range(10)))
        note = engine_for(terminal).panel_view()
        bar.update()

    # Assert
    assert note == ["line 0", "line 1", (("... 8 more lines", FAINT),)]


def test_structured_panel_draws_a_bold_borderless_table(
    terminal: FakeTerminal,
) -> None:
    # Arrange
    options = Options(stream=terminal, panel_justify="right")

    # Act
    with Bar(total=1, desc="job", options=options) as bar:
        bar.panel([{"model": "llama", "acc": 0.91234}, {"model": "gpt2", "acc": 0.5}])
        with engine_lock():
            engine_for(terminal).draw(time.monotonic())
            screen = screen_text(terminal)
        bar.update()

    # Assert
    assert screen[1:4] == [
        f"{BRAILLE.gutter}model     acc",
        f"{BRAILLE.gutter}llama  0.9123",
        f"{BRAILLE.gutter} gpt2  0.5000",
    ]
    assert "\x1b[1mmodel     acc\x1b[0m" in terminal.getvalue()


def test_live_bar_keeps_its_width_when_status_narrows(terminal: FakeTerminal) -> None:
    # Arrange
    percent_columns = []

    # Act
    with Bar(total=2, desc="job", options=Options(stream=terminal)) as bar:
        for value in ("1234", "1"):
            bar.status(x=value)
            with engine_lock():
                engine_for(terminal).draw(time.monotonic())
                percent_columns.append(screen_text(terminal)[0].index("%"))
            bar.update()

    # Assert
    assert percent_columns[0] == percent_columns[1]


def test_panel_layout_runs_without_the_lock(terminal: FakeTerminal) -> None:
    # Arrange
    acquired: list[bool] = []

    def try_lock() -> None:
        acquired.append(_engine.LOCK.acquire(timeout=1))
        if acquired[-1]:
            _engine.LOCK.release()

    class Probe:
        def __str__(self) -> str:
            other = threading.Thread(target=try_lock)
            other.start()
            other.join()
            return "probe"

    # Act
    with Bar(total=1, options=Options(stream=terminal)) as bar:
        bar.panel({"value": Probe()})
        bar.update()

    # Assert
    assert acquired == [True]


def test_panel_above_keeps_order(terminal: FakeTerminal) -> None:
    # Arrange
    options = Options(stream=terminal, panel_position="above")

    # Act
    with Bar(total=1, desc="job", options=options) as bar:
        bar.log("note")
        bar.update()

    # Assert
    lines = screen_text(terminal)
    assert lines[0] == "  info  note"
    assert "job" in lines[1]


def test_panel_above_keeps_order_over_tall_bar(terminal: FakeTerminal) -> None:
    # Arrange
    options = Options(stream=terminal, panel_position="above", height=3)

    # Act
    with Bar(total=1, desc="job", options=options) as bar:
        bar.log("note")
        bar.update()

    # Assert
    lines = screen_text(terminal)
    assert lines[0] == "  info  note"
    assert "job" in lines[1]
    assert all(lines[row].strip() for row in (2, 3))


def test_write_prints_permanently_above(terminal: FakeTerminal) -> None:
    # Act
    with Bar(total=1, desc="job", options=Options(stream=terminal)) as bar:
        bar.write("permanent")
        bar.update()

    # Assert
    assert screen_text(terminal)[0] == "permanent"


def test_logged_lines_keep_their_colored_level_above_the_bars(
    terminal: FakeTerminal,
) -> None:
    # Arrange
    options = Options(stream=terminal, panel_lines=0)

    # Act
    with Bar(total=1, desc="job", options=options) as bar:
        bar.log("disk almost full", level=logging.WARNING)
        bar.write("permanent")
        bar.update()
        backend = cast("_engine.LiveBackend", engine_for(terminal).backend)
        depth = backend.style.depth

    # Assert
    tag = to_ansi([("warn ", Ink(DEFAULT_PALETTE.stalled, bold=True))], depth)
    assert depth != ColorDepth.NONE
    assert tag in terminal.getvalue()
    assert screen_text(terminal)[:2] == ["  warn  disk almost full", "permanent"]


def test_nested_hidden_and_threaded_bars_log_in_order_into_one_panel(
    terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(_bar, "log_stamp", lambda when: "09:30:00")
    hidden = Options(stream=terminal, disable=True, log_time=True)

    def worker() -> None:
        with Bar(total=1, desc="thread", options=Options(stream=terminal)) as bar:
            bar.log("from a thread", level=logging.ERROR)
            bar.update()

    # Act
    epochs = track(range(2), "train", options=Options(stream=terminal))
    for epoch in epochs:
        batches = track(range(2), f"epoch {epoch}", options=Options(stream=terminal))
        for _ in batches:
            batches.log(f"inner {epoch}", level=logging.WARNING)
        epochs.log(f"outer {epoch}")
        epochs.write(f"saved {epoch}")
        if epoch == 1:
            with Bar(total=1, options=hidden) as helper:
                helper.log("from a hidden bar", level=logging.DEBUG)
                helper.update()
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()

    # Assert
    screen = [line for line in screen_text(terminal) if line and "%" not in line]
    assert screen == [
        "  warn  inner 0",
        "  warn  inner 0",
        "  info  outer 0",
        "saved 0",
        "  warn  inner 1",
        "  warn  inner 1",
        "  info  outer 1",
        "saved 1",
        "  09:30:00 debug from a hidden bar",
        "  error from a thread",
    ]


def test_disabled_bars_route_text_into_the_visible_panel(
    terminal: FakeTerminal,
) -> None:
    # Act
    with Bar(total=1, desc="shown", options=Options(stream=terminal)) as shown:
        hidden = Bar(
            total=1, desc="hidden", options=Options(stream=terminal, disable=True)
        )
        hidden.log("from hidden")
        routed = engine_for(terminal).panel_view()
        hidden.close()
        shown.update()

    # Assert
    assert _plain(routed) == ["info  from hidden"]
    assert "hidden" not in terminal.getvalue().replace("from hidden", "")


def test_logging_handlers_are_retargeted_and_restored(
    terminal: FakeTerminal, logger: logging.Logger
) -> None:
    # Arrange
    handler = logging.StreamHandler(sys.stderr)
    original = handler.stream
    logger.addHandler(handler)

    # Act
    with Bar(total=1, desc="job", options=Options(stream=terminal)) as bar:
        logger.warning("captured warning")
        late = logging.StreamHandler()
        logger.addHandler(late)
        bar.update()

    # Assert
    assert handler.stream is original
    assert not isinstance(late.stream, _engine.Capture)
    assert "captured warning" in screen_text(terminal)


def test_logging_from_threads_while_closing_does_not_deadlock(
    terminal: FakeTerminal, logger: logging.Logger
) -> None:
    # Arrange
    logger.addHandler(logging.StreamHandler(sys.stderr))
    stop = threading.Event()

    def spam() -> None:
        while not stop.is_set():
            logger.warning("tick")

    def open_and_close() -> None:
        for _ in range(20):
            Bar(total=1, options=Options(stream=terminal)).close()

    worker = threading.Thread(target=spam)
    closer = threading.Thread(target=open_and_close)

    # Act
    worker.start()
    closer.start()
    closer.join(timeout=10)
    finished = not closer.is_alive()
    stop.set()
    worker.join(timeout=10)

    # Assert
    assert finished
    assert not worker.is_alive()


def test_untrusted_text_cannot_inject_escapes(terminal: FakeTerminal) -> None:
    # Arrange
    evil = "\x1b]52;c;cHduZWQ=\x07\x1b[2J\x9b6n\x1b]8;;unterminated\x07"

    # Act
    with Bar(total=1, desc=evil, options=Options(stream=terminal)) as bar:
        bar.log(evil)
        bar.panel(evil)
        bar.status(**{evil: evil})
        print(evil, end="")
        bar.update()

    # Assert: the strict screen model rejects any sequence psnek never emits
    screen_text(terminal)
    output = terminal.getvalue()
    assert "\x9b" not in output
    assert "\x07" not in output
    assert "\x1b]" not in output


def test_region_is_clamped_to_terminal_height(
    terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setenv("LINES", "6")
    monkeypatch.setenv("COLUMNS", "20")

    # Act
    bars = [
        Bar(total=1, desc=f"bar {i}", options=Options(stream=terminal))
        for i in range(1200)
    ]
    with engine_lock():
        engine_for(terminal).draw(time.monotonic())
    during = screen_text(terminal)
    for bar in reversed(bars):
        bar.close()

    # Assert
    assert len(during) == 5
    assert during[-1] == "  +1196 more bars"


_OSC_CASES = [
    (Frame(n=1, total=4), "\x1b]9;4;1;25\x1b\\"),
    (Frame(n=1), "\x1b]9;4;3;0\x1b\\"),
    (Frame(n=1, total=4, state="stalled"), "\x1b]9;4;4;25\x1b\\"),
    (Frame(n=1, total=4, state="stuck"), "\x1b]9;4;2;25\x1b\\"),
    (Frame(n=5, state="done"), "\x1b]9;4;1;100\x1b\\"),
    (Frame(n=995, total=1000), "\x1b]9;4;1;99\x1b\\"),
]


def test_progress_osc_states(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv("PSNEK_OSC_PROGRESS", "1")
    backend = _engine.LiveBackend(FakeTerminal())

    # Act
    emitted = [backend._progress_osc([frame], 0.0) for frame, _ in _OSC_CASES]

    # Assert
    assert emitted == [sequence for _, sequence in _OSC_CASES]


def test_progress_osc_is_emitted_and_cleared(
    terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setenv("PSNEK_OSC_PROGRESS", "1")

    # Act
    with Bar(total=4, options=Options(stream=terminal)) as bar:
        with engine_lock():
            engine_for(terminal).draw(time.monotonic())
        bar.update(4)

    # Assert
    output = terminal.getvalue()
    screen_text(terminal)
    assert "\x1b]9;4;1;0\x1b\\" in output
    assert output.endswith("\x1b]9;4;0;0\x1b\\")


_NOTICE = re.compile(r"\x1b\]777;notify;([^;\x1b]*);")


def test_notification_sequences_follow_each_protocol() -> None:
    # Act
    kitty = _engine._notification("osc99", "job done", "1/1")

    # Assert
    assert _engine._notification("osc9", "a;b done", "1/1") == (
        "\x1b]9;a,b done: 1/1\x1b\\"
    )
    assert _engine._notification("osc777", "job done", "x;y") == (
        "\x1b]777;notify;job done;x;y\x1b\\"
    )
    assert re.fullmatch(
        r"\x1b\]99;(i=psnek\d+):d=0:o=unfocused;job done\x1b\\"
        r"\x1b\]99;\1:p=body;1/1\x1b\\",
        kitty,
    )
    assert _engine._notification("bell", "job done", "1/1") == "\a"


def test_a_failure_through_nested_bars_notifies_once(
    terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setenv("PSNEK_NOTIFY", "osc777")
    monkeypatch.setenv("PSNEK_NOTIFY_AFTER", "0")
    options = Options(stream=terminal)

    # Act
    with (
        pytest.raises(RuntimeError),
        Bar(total=4, desc="outer", options=options),
        Bar(total=2, desc="inner", options=options) as inner,
    ):
        inner.update()
        raise RuntimeError("upstream closed")

    # Assert
    screen_text(terminal)
    assert _NOTICE.findall(terminal.getvalue()) == ["inner failed"]


@pytest.mark.parametrize(
    ("after", "interrupt", "expected"),
    [("0", False, ["job done"]), ("30", False, []), ("0", True, [])],
)
def test_a_finished_session_notifies_after_the_threshold_unless_interrupted(
    terminal: FakeTerminal,
    monkeypatch: pytest.MonkeyPatch,
    after: str,
    interrupt: bool,
    expected: list[str],
) -> None:
    # Arrange
    monkeypatch.setenv("PSNEK_NOTIFY", "osc777")
    monkeypatch.setenv("PSNEK_NOTIFY_AFTER", after)

    # Act
    with Bar(total=2, desc="job", options=Options(stream=terminal)) as bar:
        bar.update()
        if interrupt:
            bar.interrupt()
        else:
            bar.update()

    # Assert
    screen_text(terminal)
    assert _NOTICE.findall(terminal.getvalue()) == expected


@pytest.mark.parametrize("notify", [True, False])
def test_a_bar_notifies_each_time_it_gets_stuck_unless_silenced(
    terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch, notify: bool
) -> None:
    # Arrange
    monkeypatch.setenv("PSNEK_NOTIFY", "osc777")
    options = Options(stream=terminal, notify=notify)

    # Act
    with Bar(total=10, desc="crawl", options=options) as bar:
        engine = engine_for(terminal)
        with engine_lock():
            take_steps(bar, bar._pace.changed_at)
        for _ in range(2):
            start = bar._pace.changed_at
            with engine_lock():
                for offset in (0.5, 32, 40):
                    engine.draw(start + offset)
            bar.update()
            with engine_lock():
                engine.draw(bar._pace.changed_at + 0.1)
        bar.update(5)

    # Assert
    screen_text(terminal)
    expected = ["crawl stuck"] * 2 if notify else []
    assert _NOTICE.findall(terminal.getvalue()) == expected


def test_silenced_bars_neither_notify_nor_use_up_the_failure_alert(
    terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setenv("PSNEK_NOTIFY", "osc777")
    monkeypatch.setenv("PSNEK_NOTIFY_AFTER", "0")
    silenced = Options(stream=terminal, notify=False)

    # Act
    with (
        pytest.raises(RuntimeError),
        Bar(total=4, desc="outer", options=Options(stream=terminal)),
        Bar(total=2, desc="inner", options=silenced),
    ):
        raise RuntimeError("upstream closed")
    with Bar(total=1, desc="quiet", options=silenced) as bar:
        bar.update()

    # Assert
    screen_text(terminal)
    assert _NOTICE.findall(terminal.getvalue()) == ["outer failed"]


def test_write_failure_disables_output_and_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    monkeypatch.setenv("PSNEK_MODE", "live")
    stream = FailingStream(BrokenPipeError("pipe closed"))

    # Act
    with Bar(total=1, options=Options(stream=stream)) as bar:
        print("after failure")
        bar.update()

    # Assert
    assert (
        "psnek output disabled after a failure: BrokenPipeError: pipe closed"
        in caplog.text
    )


def test_streams_on_one_terminal_share_a_live_region(
    terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    same_device = os.stat_result((0,) * 10)
    monkeypatch.setattr(_engine, "_file_of", lambda stream: same_device)
    other = FakeTerminal()

    # Act
    with Bar(total=1, desc="outer", options=Options(stream=terminal)) as outer:
        with Bar(total=1, desc="inner", options=Options(stream=other)) as inner:
            with engine_lock():
                engine_for(terminal).draw(time.monotonic())
            inner.update()
        outer.update()

    # Assert
    assert other.getvalue() == ""
    assert "inner" in terminal.getvalue()


def test_rendering_errors_disable_output_without_killing_the_bar(
    terminal: FakeTerminal,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    def explode(frame: Frame, width: int, style: object, label: int = 0) -> None:
        raise ValueError("bad frame")

    with Bar(total=2, options=Options(stream=terminal)) as bar:
        monkeypatch.setattr(_engine, "layout", explode)

        # Act
        with engine_lock():
            engine_for(terminal).draw(time.monotonic())
        bar.update(2)

    # Assert
    assert bar.n == 2
    assert bar.outcome == "done"
    assert "bad frame" in caplog.text
    assert terminal.getvalue().endswith("\x1b[?7h\x1b[?25h")


def test_unencodable_text_is_escaped(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.setenv("PSNEK_MODE", "log")
    raw = io.BytesIO()

    # Act
    with io.TextIOWrapper(raw, encoding="ascii") as stream:
        with Bar(total=1, desc="café", options=Options(stream=stream)) as bar:
            bar.update()
        written = raw.getvalue()

    # Assert
    assert b"caf\\xe9: 100.0%" in written


def test_log_mode_emits_plain_lines_and_a_late_note(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    stream = io.StringIO()
    monkeypatch.setenv("CI", "true")

    # Act
    with Bar(total=2, desc="ci job", options=Options(stream=stream)) as bar:
        bar.log("a log line")
        bar.panel("ephemeral status")
        bar.update(2)

    # Assert
    lines = stream.getvalue().splitlines()
    assert lines[0].startswith("ci job: 0.0% 0/2")
    assert lines[1] == "info  a log line"
    assert lines[2].startswith("ci job: 100.0% 2/2")
    assert "done" in lines[2]
    assert lines[3:] == ["ephemeral status"]
    assert "\x1b" not in stream.getvalue()
    assert "\r" not in stream.getvalue()


def test_log_mode_reports_health_changes_and_changed_panel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    stream = io.StringIO()
    monkeypatch.setenv("PSNEK_MODE", "log")
    bar = Bar(total=10, desc="job", options=Options(stream=stream))
    bar.panel("waiting on shard 3")
    engine = engine_for(stream)

    # Act: the lock keeps the render thread from drawing at real time
    with engine_lock():
        later = take_steps(bar, bar._pace.changed_at) + 20
        engine.draw(later)
        engine.draw(later + 0.5)
        bar.update(7)
        bar.close()

    # Assert
    lines = stream.getvalue().splitlines()
    assert "stalled" in lines[1]
    assert lines[2] == "waiting on shard 3"
    assert len(lines) == 4


def test_log_mode_prints_an_unchanged_panel_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    stream = io.StringIO()
    monkeypatch.setenv("PSNEK_MODE", "log")
    bar = Bar(total=10, desc="job", options=Options(stream=stream))
    engine = engine_for(stream)

    # Act: equal content set again between status lines is not news
    with engine_lock():
        later = time.monotonic() + 40
        bar.panel({"shard": 3, "state": "waiting"})
        engine.draw(later)
        bar.panel({"shard": 3, "state": "waiting"})
        engine.draw(later + 20)
        bar.update(10)
        bar.close()

    # Assert
    output = stream.getvalue()
    assert output.count("shard  3\nstate  waiting\n") == 1
    assert "\x1b" not in output


def test_log_mode_nested_bars_report_only_health_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    stream = io.StringIO()
    monkeypatch.setenv("PSNEK_MODE", "log")
    outer = Bar(total=2, desc="outer", options=Options(stream=stream))
    inner = Bar(total=5, desc="inner", options=Options(stream=stream))
    engine = engine_for(stream)

    # Act
    with engine_lock():
        start = take_steps(inner, inner._pace.changed_at)
        engine.draw(start + 1)
        quiet = "inner" not in stream.getvalue()
        engine.draw(start + 20)
        inner.update(2)
        engine.draw(start + 41)
        inner.close()
        outer.update(2)
        outer.close()

    # Assert
    inner_lines = [line for line in stream.getvalue().splitlines() if "inner" in line]
    assert quiet
    assert len(inner_lines) == 2
    assert "stalled" in inner_lines[0]
    assert inner_lines[1].endswith("(recovered)")


def test_a_bar_outliving_the_top_bar_takes_its_place(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    stream = io.StringIO()
    monkeypatch.setenv("PSNEK_MODE", "log")
    first = Bar(total=1, desc="first", options=Options(stream=stream))
    second = Bar(total=1, desc="second", options=Options(stream=stream))

    # Act
    first.update()
    first.close()
    second.update()
    second.close()

    # Assert
    assert "second: 100.0%" in stream.getvalue()


def test_notebook_uses_one_updated_output(fake_ipython: list[FakeHandle]) -> None:
    # Arrange
    stream = io.StringIO()

    # Act
    for _ in track(range(3), "<b>nb</b>", options=Options(stream=stream)):
        for _ in track(range(2), "inner", options=Options(stream=stream)):
            pass
    settle_idle_engines()

    # Assert
    assert len(fake_ipython) == 1
    final = fake_ipython[0].updates[-1]
    assert "&lt;b&gt;nb&lt;/b&gt;" in final["text/html"]
    assert "<b>nb</b>" in final["text/plain"]
    assert "100.0%" in final["text/plain"]
    assert "inner" not in final["text/plain"]


def test_notebook_text_stays_in_the_cell_output_around_the_bars(
    fake_ipython: list[FakeHandle],
) -> None:
    # Arrange
    stream = io.StringIO()

    # Act
    with Bar(total=1, desc="job", options=Options(stream=stream)) as bar:
        capturing = isinstance(sys.stdout, _engine.Capture)
        bar.log("logged")
        bar.write("written")
        bar.panel("note")
        bar.log("shard lost\nretrying", level=logging.WARNING)
        bar.log("schema", level=logging.DEBUG)
        with engine_lock():
            engine_for(stream).draw(time.monotonic())
        live = fake_ipython[0].updates[-1]
        bar.update()
    settle_idle_engines()

    # Assert
    final = fake_ipython[0].updates[-1]
    assert not capturing
    assert stream.getvalue() == ""
    assert shown_text(live).splitlines()[:2] == ["  info  logged", "written"]
    assert shown_text(live).splitlines()[3:] == [
        "⡇ note",
        "⡇ warn  shard lost",
        "⡇       retrying",
        "⡇ debug schema",
    ]
    assert final["text/plain"].splitlines()[:2] == ["  info  logged", "written"]
    assert final["text/plain"].splitlines()[2].endswith("done")
    assert final["text/plain"].splitlines()[3:] == [
        "| note",
        "  warn  shard lost",
        "        retrying",
        "  debug schema",
    ]
    assert 'color:#ba7d0a;font-weight:bold">warn </span>' in final["text/html"]
    assert '<span style="opacity:.6">debug schema</span>' in final["text/html"]


def test_a_running_notebook_output_sends_only_the_recent_lines_as_html(
    fake_ipython: list[FakeHandle],
) -> None:
    # Arrange
    stream = io.StringIO()
    count = NOTEBOOK_LIVE_LINES + 5

    # Act
    with Bar(total=1, desc="job", options=Options(stream=stream)) as bar:
        for index in range(count):
            bar.write(f"line {index}")
        with engine_lock():
            engine_for(stream).draw(time.monotonic())
        bar.update()
    settle_idle_engines()

    # Assert: only the final update carries plain text, with every line
    *updates, final = fake_ipython[0].updates
    live = shown_text(updates[-1]).splitlines()
    assert all("text/plain" not in update for update in updates)
    assert live[0] == "  +5 earlier lines"
    assert live[1 : 1 + NOTEBOOK_LIVE_LINES] == [
        f"line {index}" for index in range(5, count)
    ]
    lines = final["text/plain"].splitlines()
    assert lines[:count] == [f"line {index}" for index in range(count)]
    assert final["text/html"].count("white-space:pre-wrap") == 1


def test_notebook_panel_table_has_a_bold_header(
    fake_ipython: list[FakeHandle],
) -> None:
    # Arrange
    stream = io.StringIO()

    # Act
    with Bar(total=1, desc="job", options=Options(stream=stream)) as bar:
        bar.panel([{"model": "<b>", "acc": 0.5}])
        with engine_lock():
            engine_for(stream).draw(time.monotonic())
        bar.update()

    # Assert
    shown = next(u for u in fake_ipython[0].updates if "model" in shown_text(u))
    assert '<span style="font-weight:bold">model  acc</span>' in shown["text/html"]
    assert "&lt;b&gt;    0.5000" in shown["text/html"]
    assert "<b>    0.5000" in shown_text(shown)


def test_notebook_bar_keeps_its_width_when_status_narrows(
    fake_ipython: list[FakeHandle],
) -> None:
    # Arrange
    stream = io.StringIO()
    percent_columns = []

    # Act
    with Bar(total=2, desc="job", options=Options(stream=stream)) as bar:
        for value in ("a" * 20, "1"):
            bar.status(x=value)
            with engine_lock():
                engine_for(stream).draw(time.monotonic())
            percent_columns.append(shown_text(fake_ipython[0].updates[-1]).index("%"))
            bar.update()

    # Assert
    assert percent_columns[0] == percent_columns[1]


@pytest.mark.parametrize("clear", [False, True])
def test_notebook_keeps_each_sessions_panel_note_after_its_bars(
    fake_ipython: list[FakeHandle], clear: bool
) -> None:
    # Arrange
    options = Options(stream=io.StringIO(), panel_clear=clear)

    # Act: two loops in one cell, the first with a note
    with Bar(total=1, desc="first", options=options) as first:
        first.panel({"loss": 0.5})
        first.update()
    settle_idle_engines()
    with Bar(total=1, desc="second", options=options) as second:
        second.update()
    settle_idle_engines()

    # Assert
    rows = fake_ipython[0].updates[-1]["text/plain"].splitlines()
    assert "first" in rows[0]
    if clear:
        assert len(rows) == 2
    else:
        assert rows[1] == f"{ASCII.gutter}loss  0.5000"
    assert "second" in rows[-1]


def test_notebook_keeps_a_panel_drawn_above_above_its_sessions_bars(
    fake_ipython: list[FakeHandle],
) -> None:
    # Arrange
    above = Options(stream=io.StringIO(), panel_position="above")

    # Act: two loops in one cell, the first drawing its note above its bars
    with Bar(total=1, desc="first", options=above) as first:
        first.panel({"loss": 0.5})
        with Bar(total=1, desc="inner", options=Options(leave=True)) as inner:
            inner.update()
        first.update()
    settle_idle_engines()
    with Bar(total=1, desc="second", options=Options(stream=io.StringIO())) as second:
        second.update()
    settle_idle_engines()

    # Assert
    rows = fake_ipython[0].updates[-1]["text/plain"].splitlines()
    assert rows[0] == f"{ASCII.gutter}loss  0.5000"
    assert "inner" in rows[1]
    assert "first" in rows[2]
    assert "second" in rows[3]


def test_notebook_sessions_in_one_cell_share_one_output(
    fake_ipython: list[FakeHandle], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setenv("NO_COLOR", "1")
    stream = io.StringIO()
    options = Options(stream=stream, panel_position="above")

    # Act
    with Bar(total=1, desc="first", options=options) as bar:
        bar.panel("panel text")
        bar.update()
        with engine_lock():
            engine_for(stream).draw(time.monotonic())
    with Bar(total=1, desc="second", options=Options(stream=sys.stdout)) as bar:
        bar.update()
    settle_idle_engines()

    # Assert
    assert len(fake_ipython) == 1
    shown = next(
        update
        for update in fake_ipython[0].updates
        if "panel text" in shown_text(update)
    )
    assert shown_text(shown).index("panel text") < shown_text(shown).index("first")
    assert "color:" not in shown["text/html"]
    final = fake_ipython[0].updates[-1]["text/plain"]
    assert "first" in final
    assert "second" in final


@pytest.fixture
def shell(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Install a stand-in IPython shell whose execution count tests can advance."""
    instance = types.SimpleNamespace(execution_count=1)
    module = types.ModuleType("IPython")
    monkeypatch.setattr(module, "get_ipython", lambda: instance, raising=False)
    monkeypatch.setitem(sys.modules, "IPython", module)
    return instance


def test_each_notebook_cell_gets_its_own_output(
    fake_ipython: list[FakeHandle], shell: types.SimpleNamespace
) -> None:
    # Arrange
    options = Options(stream=io.StringIO())

    # Act
    with Bar(total=1, desc="cell one", options=options) as bar:
        bar.update()
    shell.execution_count = 2
    with Bar(total=1, desc="cell two", options=options) as bar:
        bar.update()

    # Assert
    assert len(fake_ipython) == 2
    assert "cell one" not in shown_text(fake_ipython[1].updates[-1])


def test_a_bar_left_open_in_an_earlier_cell_stays_there(
    fake_ipython: list[FakeHandle], shell: types.SimpleNamespace
) -> None:
    # Arrange
    options = Options(stream=io.StringIO())
    abandoned = Bar(total=5, desc="cell one", options=options)
    with engine_lock():
        engine_for(cast("io.StringIO", options.stream)).draw(time.monotonic())

    # Act
    shell.execution_count = 2
    with Bar(total=1, desc="cell two", options=options) as bar:
        bar.update()
    abandoned.close()

    # Assert
    assert len(fake_ipython) == 2
    assert "cell two" not in shown_text(fake_ipython[0].updates[-1])
    assert "cell one" not in shown_text(fake_ipython[1].updates[-1])


def test_interleaved_notebook_requests_keep_their_own_outputs(
    fake_ipython: list[FakeHandle], shell: types.SimpleNamespace
) -> None:
    # Arrange
    request = {"header": {"msg_id": "cell-a", "msg_type": "execute_request"}}
    shell.kernel = types.SimpleNamespace(get_parent=lambda: request)
    options = Options(stream=io.StringIO())
    descs = ("a first", "b only", "a second")

    # Act
    for desc in descs:
        request["header"]["msg_id"] = f"cell-{desc[0]}"
        with Bar(total=1, desc=desc, options=options) as bar:
            bar.update()
        settle_idle_engines()

    # Assert
    assert len(fake_ipython) == 2
    final = [handle.updates[-1]["text/plain"] for handle in fake_ipython]
    assert "a first" in final[0]
    assert "a second" in final[0]
    assert "a second" not in final[1]


def test_notebook_text_for_stderr_bars_goes_to_stdout(
    fake_ipython: list[FakeHandle], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    stdout, stderr = io.StringIO(), io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(_engine, "in_notebook", lambda: True)

    # Act
    with Bar(total=1, desc="job", options=Options(disable=True)) as bar:
        bar.log("logged")
        bar.update()

    # Assert
    assert stdout.getvalue() == "info  logged\n"
    assert stderr.getvalue() == ""


@pytest.mark.usefixtures("always_busy")
def test_notebook_final_frame_is_drawn_before_the_cell_ends(
    fake_ipython: list[FakeHandle], cell_events: FakeEvents
) -> None:
    # Arrange
    stream = io.StringIO()

    # Act: the lock keeps the render thread from drawing the deferred frame
    with engine_lock():
        for _ in range(2):
            with Bar(total=3, desc="job", options=Options(stream=stream)) as bar:
                bar.update(3)
        deferred = shown_text(fake_ipython[0].updates[-1])
        cell_events.end_cell()

    # Assert: registered once, and the cell's output ends on the final frame
    assert len(cell_events.callbacks["post_run_cell"]) == 1
    assert "done" not in deferred
    assert "done" in fake_ipython[0].updates[-1]["text/plain"]


def test_notebook_batches_bar_churn_into_ticks(
    fake_ipython: list[FakeHandle],
) -> None:
    # Arrange
    options = Options(stream=io.StringIO())

    # Act
    with Bar(total=50, desc="outer", options=options) as outer:
        for _ in range(50):
            Bar(total=1, desc="inner", options=options).close()
            outer.update()

    # Assert: only the closing update, not two per inner bar
    assert sum(len(handle.updates) for handle in fake_ipython) <= 3


def test_notebook_keeps_a_bounded_number_of_rows(
    fake_ipython: list[FakeHandle],
) -> None:
    # Arrange
    stream = io.StringIO()

    # Act
    with Bar(total=1, desc="outer", options=Options(stream=stream)) as outer:
        for index in range(_engine.NOTEBOOK_KEPT + 3):
            child = Bar(
                total=1,
                desc=f"kept {index}",
                options=Options(stream=stream, leave=True),
            )
            child.update()
            child.close()
        printed_while_open = stream.getvalue()
        outer.update()
    settle_idle_engines()

    # Assert
    final = fake_ipython[0].updates[-1]["text/plain"].splitlines()
    assert printed_while_open == ""
    assert final[0] == "  +4 earlier bars"
    assert len(final) == _engine.NOTEBOOK_KEPT + 1


def test_notebook_display_failure_is_logged(
    fake_ipython: list[FakeHandle],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    def display(obj: dict[str, str], *, raw: bool, display_id: bool) -> None:
        raise RuntimeError("no frontend")

    module = types.ModuleType("IPython.display")
    monkeypatch.setattr(module, "display", display, raising=False)
    monkeypatch.setitem(sys.modules, "IPython.display", module)

    stream = io.StringIO()

    # Act
    with Bar(total=1, options=Options(stream=stream)) as bar:
        bar.update()
        bar.write("still printed")

    # Assert
    assert "psnek notebook display disabled: RuntimeError: no frontend" in caplog.text
    assert stream.getvalue() == "still printed\n"


def test_an_engine_lingers_for_the_next_bar_then_stops(
    terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    with Bar(total=1, options=Options(stream=terminal)) as bar:
        engine = engine_for(terminal)
        stop = engine.stop_event
        bar.update()
    restored = terminal.getvalue().endswith(_engine.SHOW_CURSOR)

    # Act
    with Bar(total=1, options=Options(stream=terminal)) as bar:
        reused = engine_for(terminal) is engine and engine.stop_event is stop
        bar.update()
    monkeypatch.setattr(_engine, "LINGER", 0.0)
    stopped = stop.wait(5)

    # Assert
    assert restored
    assert reused
    assert stopped
    assert (id(terminal), False) not in _engine.ENGINES


def test_bars_closed_by_the_collector_in_a_critical_section_close_afterwards(
    terminal: FakeTerminal,
) -> None:
    # Arrange
    bar = track(iter(range(5)), options=Options(stream=terminal))
    iterator = iter(bar)
    next(iterator)
    engine = engine_for(terminal)

    # Act
    with engine_lock():
        del iterator
        open_during_draw = bar.outcome is None
    _deferred.run_deferred()

    # Assert
    assert open_during_draw
    assert bar.outcome == "interrupted"
    assert not engine.sources


def _exit_code(pid: int, timeout: float = 10.0) -> int:
    """Wait for a forked child, killing it if it hangs."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.01)
    os.kill(pid, signal.SIGKILL)
    os.waitpid(pid, 0)
    pytest.fail("the forked child hung")


@pytest.mark.integration
@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
@pytest.mark.filterwarnings(
    "ignore:This process .* is multi-threaded:DeprecationWarning"
)
def test_forked_child_restores_streams_and_forgets_running_cells(
    terminal: FakeTerminal,
) -> None:
    # Arrange
    bar = Bar(total=1, options=Options(stream=terminal))
    _backends.start_cell()

    # Act
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            healthy = (
                not isinstance(sys.stdout, _engine.Capture)
                and not _engine.ENGINES
                and not _backends._RUNNING
            )
            child = Bar(total=1, options=QUIET)
            child.update()
            child.close()
            code = 0 if healthy and child.outcome == "done" else 1
        finally:
            os._exit(code)
    code = _exit_code(pid)
    bar.update()
    bar.close()

    # Assert
    assert code == 0


def test_a_stale_capture_left_by_a_stream_swap_is_unwrapped(
    terminal: FakeTerminal,
) -> None:
    # Arrange
    with Bar(total=1, options=Options(stream=terminal)) as bar:
        stale = sys.stdout
        bar.update()
    sys.stdout = stale
    try:
        # Act
        with Bar(total=1, options=Options(stream=terminal)) as bar:
            print("routed")
            shown = engine_for(terminal).panel_view()
            bar.update()
    finally:
        sys.stdout = cast("_engine.Capture", stale).original

    # Assert
    assert shown == ["routed"]


class _EndedForeignThread:
    """Like ``threading._DummyThread`` after its OS thread ended on 3.13+."""

    def is_alive(self) -> bool:
        raise RuntimeError("thread is not alive")


def test_partial_text_of_an_ended_foreign_thread_is_reaped() -> None:
    # Arrange
    capture = _engine.Capture(io.StringIO())
    thread = cast("threading.Thread", _EndedForeignThread())
    capture.pending[thread] = io.StringIO("partial")

    # Act
    reaped = capture.reap()

    # Assert
    assert reaped == ["partial"]
    assert not capture.pending


def test_partial_lines_of_finished_threads_move_to_the_log(
    terminal: FakeTerminal,
) -> None:
    # Arrange
    with Bar(total=1, options=Options(stream=terminal)) as bar:
        worker = threading.Thread(target=lambda: print("half a line", end=""))
        worker.start()
        worker.join()
        engine = engine_for(terminal)

        # Act
        with engine_lock():
            engine.draw(time.monotonic())
            logged = list(engine.logs)
            pending = sum(len(capture.pending) for capture in _engine.CAPTURES.values())
        bar.update()

    # Assert
    assert logged == ["half a line"]
    assert pending == 0


def test_kept_nested_bars_print_after_earlier_logs(terminal: FakeTerminal) -> None:
    # Arrange
    options = Options(stream=terminal)
    with Bar(total=1, desc="outer", options=options) as outer:
        outer.log("logged first")

        # Act
        with Bar(total=1, desc="inner", options=Options(stream=terminal, leave=True)):
            pass
        outer.update()

    # Assert
    lines = screen_text(terminal)
    inner = next(i for i, line in enumerate(lines) if "inner" in line)
    assert lines.index("  info  logged first") < inner


def test_kept_lines_align_with_the_live_bars(terminal: FakeTerminal) -> None:
    # Arrange
    options = Options(stream=terminal)
    with Bar(total=1000, desc="outer", options=options) as outer:
        outer.update(500)
        with Bar(total=8, desc="inner", options=Options(stream=terminal, leave=True)):
            pass

        # Act
        with engine_lock():
            engine_for(terminal).draw(time.monotonic())
        during = screen_text(terminal)
        outer.update(500)

    # Assert
    kept = next(line for line in during if "inner" in line)
    live = next(line for line in during if "outer" in line)
    assert kept.index("%") == live.index("%")


def test_text_is_still_attempted_after_a_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    monkeypatch.setenv("PSNEK_MODE", "live")
    stream = FlakyTerminal()
    with Bar(total=1, options=Options(stream=stream)) as bar:
        bar.log("waiting in the panel")
        stream.fail_next = True
        with engine_lock():
            engine_for(stream).draw(time.monotonic() + 1)

        # Act
        bar.write("after the failure")
        bar.update()

    # Assert
    output = stream.getvalue()
    assert "waiting in the panel\n" in output
    assert "after the failure\n" in output
    assert "psnek output disabled" in caplog.text


def test_log_mode_announces_promoted_bars_and_interrupted_nested_bars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    stream = io.StringIO()
    monkeypatch.setenv("PSNEK_MODE", "log")
    options = Options(stream=stream)
    first = Bar(total=1, desc="first", options=options)
    second = Bar(total=1, desc="second", options=options)
    third = Bar(total=1, desc="third", options=options)

    # Act
    first.update()
    first.close()
    with engine_lock():
        engine_for(stream).draw(time.monotonic())
    third.interrupt()
    second.update()
    second.close()

    # Assert
    lines = stream.getvalue().splitlines()
    assert lines[2].startswith("second: 0.0%")
    assert lines[3].startswith("third:")
    assert "interrupted" in lines[3]


def test_a_lingering_log_engine_starts_a_clean_session_for_the_next_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    stream = io.StringIO()
    monkeypatch.setenv("PSNEK_MODE", "log")
    options = Options(stream=stream)
    with Bar(total=1, desc="first", options=options) as bar:
        bar.update()
    engine = engine_for(stream)

    # Act
    with Bar(total=1, desc="sec\rond\x1b[31m", options=options) as bar:
        reused = engine_for(stream) is engine
        bar.update()

    # Assert
    lines = stream.getvalue().splitlines()
    assert reused
    assert [line.split(":")[0] for line in lines] == ["first"] * 2 + [
        "sec\\u000dond"
    ] * 2
    assert "\x1b" not in stream.getvalue()


class _Recorder(io.StringIO):
    def __init__(self, events: list[str], name: str) -> None:
        super().__init__()
        self.events = events
        self.name = name

    def write(self, text: str) -> int:
        self.events.append(self.name)
        return super().write(text)

    def flush(self) -> None:
        self.events.append(f"{self.name} flush")


def test_stdout_is_flushed_before_a_start_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("PSNEK_MODE", "log")
    events: list[str] = []
    monkeypatch.setattr(sys, "stdout", _Recorder(events, "stdout"))
    stream = _Recorder(events, "bar")

    # Act
    bar = Bar(total=1, desc="job", options=Options(stream=stream))
    bar.close()

    # Assert
    assert events.index("stdout flush") < events.index("bar")


def test_the_render_thread_survives_an_unexpected_error(
    terminal: FakeTerminal,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    raised = threading.Event()
    draw = _engine.Engine.draw

    def explode(self: _engine.Engine, now: float) -> None:
        if threading.current_thread().name == "psnek-render" and not raised.is_set():
            raised.set()
            raise RuntimeError("draw bug")
        draw(self, now)

    monkeypatch.setattr(_engine.Engine, "draw", explode)
    with Bar(total=1, options=Options(stream=terminal)) as bar:
        # Act
        raised.wait(5)
        engine = engine_for(terminal)
        time.sleep(3 * _engine.TICK)
        ticking = engine.ticking
        bar.update()

    # Assert
    assert raised.is_set()
    assert ticking
    assert "draw bug" in caplog.text


_OVERWRITES = {
    "日本\rxy": "xy本",
    "10%\x1b[2K\x1b[1G20%": "20%",
    "abcdef\x1b[3G\x1b[K": "ab",
    "abcdef\x1b[3G\x1b[1K": "   def",
    "abc\x1b[5Gx": "abc x",
    "日本\x1b[2Gx": " x本",
    "ab\x1b[5G\tX": "ab      X",
    "\x1b[99999999999Gx": " " * 4095 + "x",
    "a" + "\u0301" * 20 + "\x1b[K": "a" + "\u0301" * _text.MAX_MARKS,
    "\U0001f5d1\ufe0f 50%\rXY": "XY 50%",
    "\x1b]0;a\rb\x1b\\done": "done",
}


def test_overwrites_follow_terminal_cells() -> None:
    assert _engine.clean_lines(list(_OVERWRITES)) == list(_OVERWRITES.values())


def test_module_write_prints_above_the_live_bars(terminal: FakeTerminal) -> None:
    # Act
    with Bar(total=1, desc="job", options=Options(stream=terminal)) as bar:
        psnek.write("saved", 3)
        bar.update()

    # Assert
    assert screen_text(terminal)[0] == "saved 3"


@pytest.mark.integration
def test_streams_may_share_a_file_only_when_their_files_match_or_are_unknown() -> None:
    # Arrange
    read_end, write_end = os.pipe()
    other_read, other_write = os.pipe()
    with (
        open(read_end, "rb"),
        open(other_read, "rb"),
        open(write_end, "w") as target,
        open(os.dup(write_end), "w") as same,
        open(other_write, "w") as different,
    ):
        # Act
        shared = _engine._may_share_file(cast("TextIO", same), target)
        separate = _engine._may_share_file(cast("TextIO", different), target)
        unknown = _engine._may_share_file(io.StringIO(), target)
        strict_unknown = _engine._may_share_file(io.StringIO(), target, strict=True)
        both_unknown = _engine._may_share_file(
            io.StringIO(), io.StringIO(), strict=True
        )
        missing = _engine._may_share_file(None, target)
    _deferred.run_deferred()

    # Assert
    assert shared
    assert not separate
    assert unknown
    assert not missing
    assert not strict_unknown
    assert both_unknown


def test_any_stream_failure_is_logged_without_raising(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    monkeypatch.setenv("PSNEK_MODE", "log")
    stream = FailingStream(RuntimeError("custom stream bug"))

    # Act
    with Bar(total=1, options=Options(stream=stream)) as bar:
        bar.update()

    # Assert
    assert bar.outcome == "done"
    assert "custom stream bug" in caplog.text


def test_a_hidden_bar_logs_into_the_live_panel_of_the_same_terminal(
    terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    same_device = os.stat_result((0,) * 10)
    monkeypatch.setattr(_engine, "_file_of", lambda stream: same_device)
    other = FakeTerminal()

    # Act
    with Bar(total=1, options=Options(stream=terminal)) as live:
        hidden = Bar(total=1, options=Options(stream=other, disable=True))
        hidden.log("from hidden")
        shown = engine_for(terminal).panel_view()
        hidden.close()
        live.update()

    # Assert
    assert _plain(shown) == ["info  from hidden"]
    assert other.getvalue() == ""


def test_a_lingering_engine_yields_to_a_live_region_on_the_same_terminal(
    terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    same_device = os.stat_result((0,) * 10)
    monkeypatch.setattr(_engine, "_file_of", lambda stream: same_device)
    other = FakeTerminal()
    with Bar(total=1, options=Options(stream=other)) as finished:
        finished.update()

    # Act
    with Bar(total=1, options=Options(stream=terminal)) as live:
        joined = Bar(total=1, options=Options(stream=other))
        finished.log("after linger")
        shown = engine_for(terminal).panel_view()
        shared = joined._engine is live._engine
        joined.close()
        live.update()

    # Assert
    assert shared
    assert _plain(shown) == ["info  after linger"]
    assert "after linger" not in other.getvalue()


def test_an_interrupted_final_draw_still_restores_the_terminal(
    terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    draw = _engine.LiveBackend.draw

    def interrupt_last(
        backend: _engine.LiveBackend,
        engine: _engine.Engine,
        frames: list[Frame],
        now: float,
    ) -> None:
        if not engine.sources:
            engine.emit(_engine.SYNC_BEGIN)
            raise KeyboardInterrupt
        draw(backend, engine, frames, now)

    monkeypatch.setattr(_engine.LiveBackend, "draw", interrupt_last)

    # Act
    with (
        pytest.raises(KeyboardInterrupt),
        Bar(total=1, options=Options(stream=terminal)) as bar,
    ):
        bar.update()

    # Assert
    output = terminal.getvalue()
    assert output.endswith(_engine.WRAP_ON + _engine.SHOW_CURSOR)
    assert output.count(_engine.SYNC_BEGIN) == output.count(_engine.SYNC_END)
    assert engine_for(terminal).idle_since is not None


def test_exit_finishes_every_bar_even_if_one_fails_to(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    first = Bar(total=2, options=QUIET)
    second = Bar(total=2, options=QUIET)
    abandon = Bar._abandon

    def broken_for_second(bar: Bar) -> None:
        if bar is second:
            raise RuntimeError("abandon bug")
        abandon(bar)

    monkeypatch.setattr(Bar, "_abandon", broken_for_second)

    # Act
    _engine._shutdown()

    # Assert
    assert first.outcome == "interrupted"
    assert "abandon bug" in caplog.text
    second.close()


def test_a_released_capture_writes_straight_through(
    terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    original = io.StringIO()
    monkeypatch.setattr(sys, "stdout", original)
    with Bar(total=1, options=Options(stream=terminal)) as bar:
        capture = cast("_engine.Capture", sys.stdout)
        bar.update()

    # Act
    written = capture.write("after\n")

    # Assert
    assert written == len("after\n")
    assert original.getvalue() == "after\n"


def test_a_stream_replaced_during_a_bar_is_not_overwritten(
    terminal: FakeTerminal,
) -> None:
    # Arrange
    original = sys.stdout
    replacement = FakeTerminal()

    # Act
    try:
        with Bar(total=1, options=Options(stream=terminal)) as bar:
            captured = sys.stdout
            sys.stdout = replacement
            bar.update()
        after = sys.stdout
    finally:
        sys.stdout = original

    # Assert
    assert isinstance(captured, _engine.Capture)
    assert after is replacement


class _TerminalOnFile(FakeTerminal):
    """A fake TTY backed by a real file descriptor, so it can be identified."""

    def __init__(self, descriptor: int) -> None:
        super().__init__()
        self.descriptor = descriptor

    def fileno(self) -> int:
        return self.descriptor


@pytest.mark.integration
def test_stdout_redirected_to_memory_is_not_captured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("PSNEK_MODE", "live")
    redirected = io.StringIO()
    read_end, write_end = os.pipe()
    with open(read_end, "rb"), open(write_end, "wb") as device:
        terminal = _TerminalOnFile(device.fileno())

        # Act
        with (
            contextlib.redirect_stdout(redirected),
            Bar(total=1, options=Options(stream=terminal)) as bar,
        ):
            replaced = sys.stdout is not redirected
            print("to the buffer")
            bar.update()

    # Assert
    assert not replaced
    assert redirected.getvalue() == "to the buffer\n"
    assert "to the buffer" not in terminal.getvalue()


def test_a_tail_never_starts_inside_an_escape_sequence() -> None:
    # Arrange
    cases = [
        ("a" * 100 + "\x1b[31mred", "\x1b[31mred"),
        ("a" * 100 + "bcdefg", "efg"),
        ("ab", "ab"),
    ]

    # Act
    shown = [_engine._tail(_engine._buffer(text), 3) for text, _ in cases]

    # Assert
    assert shown == [tail for _, tail in cases]


_KeptCase = tuple[float | None, int, dict[str, float]]


def _kept_line(terminal: FakeTerminal, case: _KeptCase, drawn: bool) -> str:
    total, count, status = case
    options = Options(stream=terminal)
    with engine_lock(), Bar(total=total, desc="job", options=options) as bar:
        bar.status(**status)
        if drawn:
            engine_for(terminal).draw(time.monotonic())
        bar.update(count)
    return re.sub(r" +\S+it/s", " rate", screen_text(terminal)[0])


def test_a_bar_kept_before_its_first_draw_is_laid_out_like_a_drawn_one(
    terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setenv("COLUMNS", "50")
    case: _KeptCase = (20000, 12345, {"loss": 0.5})

    # Act
    early = _kept_line(terminal, case, drawn=False)
    drawn = _kept_line(FakeTerminal(), case, drawn=True)

    # Assert
    assert early == drawn


def test_log_mode_marks_a_top_level_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    stream = io.StringIO()
    monkeypatch.setenv("PSNEK_MODE", "log")
    bar = Bar(total=10, desc="job", options=Options(stream=stream))
    engine = engine_for(stream)

    # Act
    with engine_lock():
        later = take_steps(bar, bar._pace.changed_at) + 20
        engine.draw(later)
        bar.update(7)
        engine.draw(later + 1)
        bar.close()

    # Assert
    lines = stream.getvalue().splitlines()
    assert "stalled" in lines[1]
    assert lines[2].endswith("(recovered)")


def test_a_collector_close_waits_only_a_few_ticks_for_a_lock_held_elsewhere(
    terminal: FakeTerminal,
) -> None:
    # Arrange
    bar = track(iter(range(5)), options=Options(stream=terminal))
    iterator = iter(bar)
    next(iterator)
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with engine_lock():
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait(5)

    # Act
    try:
        started = time.monotonic()
        del iterator
        waited = time.monotonic() - started
        open_while_held = bar.outcome is None
    finally:
        release.set()
        holder.join(5)
    _deferred.run_deferred()

    # Assert
    assert open_while_held
    assert waited < _engine.TICK * 4
    assert bar.outcome == "interrupted"


def test_bars_without_a_standard_error_stream_write_nowhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setattr(sys, "stderr", None)

    # Act
    with Bar(total=1, desc="headless") as bar:
        bar.log("unseen")
        bar.update()

    # Assert
    assert bar.outcome == "done"
    assert bar._engine is not None
    assert bar._engine.stream is _engine.DISCARD
    assert _engine.DISCARD.write("dropped") == len("dropped")


def test_a_failed_notebook_print_keeps_the_display_updating(
    fake_ipython: list[FakeHandle], caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    stream = FlakyTerminal()
    stream.fail_next = True

    # Act
    with Bar(total=2, desc="job", options=Options(stream=stream)) as bar:
        bar.update(2)
    settle_idle_engines()
    bar.write("lost")
    bar.write("kept")

    # Assert
    (handle,) = fake_ipython
    assert "100.0%" in handle.updates[-1]["text/plain"]
    assert stream.getvalue() == "kept\n"
    assert "psnek dropped output: OSError: transient failure" in caplog.text


def test_a_bar_ends_its_session_even_if_releasing_capture_is_interrupted(
    terminal: FakeTerminal, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    def interrupt(engine: _engine.Engine, original: TextIO, lines: list[str]) -> None:
        raise KeyboardInterrupt

    deliver = _engine._deliver
    bar = Bar(total=1, options=Options(stream=terminal))
    engine = engine_for(terminal)
    monkeypatch.setattr(_engine, "_deliver", interrupt)

    # Act
    with engine_lock(), pytest.raises(KeyboardInterrupt):
        bar.close()
    restored = terminal.getvalue().endswith(_engine.SHOW_CURSOR)
    monkeypatch.setattr(_engine, "_deliver", deliver)
    settle_idle_engines()

    # Assert
    assert bar not in engine.sources
    assert restored
    assert not _engine.ENGINES


def test_an_abandoned_loop_flushes_stdout_before_its_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("PSNEK_MODE", "log")
    stream = io.StringIO()
    bar = track(iter(range(5)), "job", options=Options(stream=stream))
    iterator = iter(bar)
    next(iterator)
    shown_at_flush: list[str] = []

    def record(target: io.StringIO, ordered: bool = False) -> None:
        if not ordered:
            shown_at_flush.append(target.getvalue())

    monkeypatch.setattr(_engine, "flush_stdout", record)

    # Act
    del iterator

    # Assert
    assert bar.outcome == "interrupted"
    assert len(shown_at_flush) == 1
    assert "interrupted" not in shown_at_flush[0]
    assert "interrupted" in stream.getvalue()


def test_text_goes_to_the_stream_that_most_recently_started_a_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("PSNEK_MODE", "log")
    first, second = io.StringIO(), io.StringIO()
    bars = [
        Bar(total=1, desc=desc, options=Options(stream=stream, capture=False))
        for desc, stream in (("a1", first), ("b1", second), ("a2", first))
    ]

    # Act
    psnek.write("routed")
    for bar in bars:
        bar.close()

    # Assert
    assert "routed" in first.getvalue()
    assert "routed" not in second.getvalue()


def test_a_render_thread_flushes_stdout_only_before_ordered_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    stdout = CountingStdout()
    monkeypatch.setattr(sys, "stdout", stdout)
    flushes: list[int] = []

    def render() -> None:
        _engine._RENDERING.active = True
        _engine.flush_stdout(io.StringIO())
        flushes.append(stdout.flushes)
        _engine.flush_stdout(io.StringIO(), ordered=True)
        flushes.append(stdout.flushes)

    # Act
    thread = threading.Thread(target=render)
    thread.start()
    thread.join()
    _engine.flush_stdout(io.StringIO())

    # Assert
    assert flushes == [0, 1]
    assert stdout.flushes == 2


def test_a_live_bar_keeps_its_width_through_a_stall_and_its_end(
    terminal: FakeTerminal,
) -> None:
    # Arrange
    bar = Bar(total=100, desc="job", options=Options(stream=terminal))
    engine = engine_for(terminal)

    def shown() -> tuple[str, int]:
        line = next(line for line in screen_text(terminal) if "job" in line)
        return line, sum(0x2800 <= ord(char) <= 0x28FF for char in line[1:])

    # Act
    with engine_lock():
        start = take_steps(bar, bar._pace.changed_at)
        engine.draw(start)
        running = shown()
        engine.draw(start + 20)
        stalled = shown()
        bar.update(100 - bar.n)
        bar.close()
    settle_idle_engines()
    done = shown()

    # Assert
    assert "stalled" in stalled[0]
    assert done[0].endswith("it/s  done")
    assert running[1] == stalled[1] == done[1]


@pytest.mark.parametrize(
    ("error", "cause"), [(KeyboardInterrupt, "Ctrl-C"), (ValueError, "ValueError")]
)
def test_a_live_bar_ended_by_an_exception_keeps_its_cause_on_screen(
    terminal: FakeTerminal, error: type[BaseException], cause: str
) -> None:
    # Arrange
    options = Options(stream=terminal)

    # Act
    with pytest.raises(error), Bar(10, "train", options=options) as bar:
        bar.update(3)
        with engine_lock():
            engine_for(terminal).draw(time.monotonic())
        raise error

    # Assert
    final = [line for line in screen_text(terminal) if line][-1]
    assert final.endswith(cause)
