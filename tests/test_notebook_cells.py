"""Tests of what a notebook cell's output keeps, in what order, and when.

Until the cell ends, updates show a trimmed view; the cell's end sends the
complete output once. These tests end cells by hand through a stand-in
IPython shell, and hold the engine lock so only their own draws send updates.
"""

import io
import os
import subprocess
import sys
import threading
import time
import types
from collections.abc import Callable

import pytest

from psnek import Bar, Options, _backends, _deferred
from psnek._backends import NOTEBOOK_KEPT, NOTEBOOK_LIVE_BARS, cell_ended, notebook_key
from tests.fakes import (
    FakeEvents,
    FakeHandle,
    engine_for,
    engine_lock,
    settle_idle_engines,
    shown_text,
)


def _final(fake_ipython: list[FakeHandle]) -> list[str]:
    """Return the rows of the last update's plain text."""
    return fake_ipython[0].updates[-1]["text/plain"].splitlines()


@pytest.mark.usefixtures("never_busy")
def test_sessions_in_a_cell_send_trimmed_ends_and_the_cell_end_all_of_it(
    fake_ipython: list[FakeHandle], cell_events: FakeEvents
) -> None:
    # Arrange
    stream = io.StringIO()

    # Act
    with engine_lock():
        for epoch in range(30):
            options = Options(stream=stream)
            with Bar(total=1, desc=f"epoch {epoch}", options=options) as bar:
                for line in range(10):
                    bar.write(f"epoch {epoch} line {line}")
                bar.update()
        before_end = list(fake_ipython[0].updates)
        cell_events.end_cell()

    # Assert: no session's end resends the whole cell, and the cell's end
    # does; a session's end is complete only while nothing is left out
    assert not [
        update
        for update in before_end
        if "text/plain" in update and "earlier" in shown_text(update)
    ]
    assert "text/plain" in before_end[1]
    assert "text/plain" not in before_end[-1]
    assert all(shown_text(update).count(" line ") <= 20 for update in before_end)
    final = _final(fake_ipython)
    assert sum(" line " in row for row in final) == 300
    assert final[-2:] == ["epoch 29 line 9", final[-1]]
    assert "epoch 29" in final[-1]


def test_a_bar_left_open_at_the_cell_end_shows_all_then_runs_on_trimmed(
    fake_ipython: list[FakeHandle],
    cell_events: FakeEvents,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: the live update is large enough to delay the next for minutes
    monkeypatch.setattr(_backends, "NOTEBOOK_RATE", 10)
    stream = io.StringIO()
    bar = Bar(total=5, desc="left open", options=Options(stream=stream))
    for line in range(30):
        bar.write(f"line {line}")
    with engine_lock():
        engine_for(stream).draw(time.monotonic())
        trimmed = fake_ipython[0].updates[-1]

        # Act
        cell_events.end_cell()
        complete = fake_ipython[0].updates[-1]
        bar.write("after the end")
        engine_for(stream).draw(time.monotonic())
        live = fake_ipython[0].updates[-1]
    bar.close()
    settle_idle_engines()

    # Assert: the complete update holds back no live one, and the session's
    # end, which no cell end follows, is complete
    assert "line 0" not in trimmed["text/html"]
    assert all(f"line {line}" in complete["text/plain"] for line in range(30))
    assert "left open" in complete["text/plain"]
    assert "after the end" in shown_text(live)
    assert "text/plain" not in live
    assert "line 0" not in live["text/html"]
    assert "after the end" in fake_ipython[0].updates[-1]["text/plain"]


def test_a_failed_display_prints_the_text_it_kept_into_the_cell(
    fake_ipython: list[FakeHandle],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    def refuse(obj: dict[str, str], *, raw: bool) -> None:
        raise RuntimeError("frontend gone")

    stream = io.StringIO()
    with engine_lock(), Bar(total=1, options=Options(stream=stream)) as bar:
        for line in range(30):
            bar.write(f"line {line}")
        engine_for(stream).draw(time.monotonic())
        bar.write("line 30")
        monkeypatch.setattr(fake_ipython[0], "update", refuse)

        # Act
        engine_for(stream).draw(time.monotonic())
        bar.write("line 31")
        bar.update()
    _deferred.run_deferred()

    # Assert: each line shows in the display or the cell, in order
    printed = stream.getvalue().splitlines()
    assert printed == [f"line {line}" for line in range(32)]
    assert "psnek notebook display disabled" in caplog.text


@pytest.mark.usefixtures("never_busy")
def test_a_note_above_goes_before_its_session_text_logs_and_bars(
    fake_ipython: list[FakeHandle],
) -> None:
    # Arrange: both sessions write the same text, and only the second's
    # moves; the first keeps no bar, so its text stays apart all the same
    stream = io.StringIO()
    above = Options(stream=stream, panel_position="above")
    kept = Options(stream=stream, leave=True)

    # Act
    with engine_lock():
        with Bar(total=1, options=Options(stream=stream, leave=False)) as first:
            first.write("---")
            first.update()
        with Bar(total=1, desc="outer", options=above) as outer:
            outer.write("---")
            outer.panel("the note")
            with Bar(total=1, desc="inner", options=kept) as inner:
                inner.update()
            outer.log("logged one")
            outer.log("logged two")
            outer.update()

    # Assert
    final = _final(fake_ipython)
    assert final[:5] == [
        "---",
        "| the note",
        "  info  logged one",
        "  info  logged two",
        "---",
    ]
    assert [row.split()[1] for row in final[5:]] == ["inner", "outer"]


@pytest.mark.usefixtures("never_busy")
def test_a_note_above_and_the_text_around_it_keep_their_place_as_bars_drop(
    fake_ipython: list[FakeHandle],
) -> None:
    # Arrange
    stream = io.StringIO()
    above = Options(stream=stream, panel_position="above")
    kept = Options(stream=stream, leave=True)

    # Act: the bars dropped include the one between the sessions' text
    with engine_lock():
        with Bar(total=1, desc="a", options=Options(stream=stream)) as first:
            first.write("text a")
            first.update()
        with Bar(total=1, desc="b", options=above) as second:
            second.write("text b")
            second.panel("the note")
            for index in range(NOTEBOOK_KEPT):
                Bar(total=1, desc=f"kept {index}", options=kept).close()
            second.update()

    # Assert: the oldest bars go, but not the note
    final = _final(fake_ipython)
    assert final[:4] == ["  +3 earlier bars", "text a", "| the note", "text b"]
    assert final[4].split()[1:3] == ["kept", "2"]


@pytest.mark.usefixtures("always_busy")
def test_a_session_closed_on_a_later_tick_keeps_its_last_logs(
    fake_ipython: list[FakeHandle],
) -> None:
    # Arrange: the first session ends right after its display was created,
    # so it closes on a later tick, which the second session comes before
    stream = io.StringIO()
    above = Options(stream=stream, panel_position="above")

    # Act
    with engine_lock():
        with Bar(total=1, desc="first", options=Options(stream=stream)) as first:
            first.log("tail of the first session")
            first.update()
        with Bar(total=1, desc="second", options=above) as second:
            second.panel("note of the second session")
            second.update()
        engine_for(stream).close_idle()

    # Assert
    final = _final(fake_ipython)
    tail = next(row for row, line in enumerate(final) if "tail of the first" in line)
    note = next(row for row, line in enumerate(final) if "note of the second" in line)
    assert tail < note


@pytest.mark.usefixtures("never_busy")
def test_only_the_latest_lines_are_kept_beyond_the_line_limit(
    fake_ipython: list[FakeHandle], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(_backends, "NOTEBOOK_KEPT_LINES", 50)
    stream = io.StringIO()
    kept = Options(stream=stream, leave=True)

    # Act: the lines span text blocks between kept bars
    outer = Bar(total=1, desc="outer", options=Options(stream=stream))
    with engine_lock(), outer:
        for block in range(4):
            for line in range(30):
                outer.write(f"line {block * 30 + line}")
            Bar(total=1, desc=f"kept {block}", options=kept).close()

    # Assert
    final = _final(fake_ipython)
    assert final[0] == "  +70 earlier lines"
    lines = [row for row in final if row.startswith("line ")]
    assert lines == [f"line {line}" for line in range(70, 120)]
    assert sum(row.split()[1] == "kept" for row in final) == 4
    assert len(_backends._OUTPUTS[next(iter(_backends._OUTPUTS))].kept) <= 9


@pytest.mark.usefixtures("never_busy")
def test_a_note_above_finds_its_place_as_the_line_limit_drops_its_text(
    fake_ipython: list[FakeHandle], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(_backends, "NOTEBOOK_KEPT_LINES", 5)
    stream = io.StringIO()
    above = Options(stream=stream, panel_position="above")
    kept = Options(stream=stream, leave=True)

    # Act: the session's first text and the logs kept with the note count
    # toward the limit, so that text is all dropped
    with engine_lock(), Bar(total=1, desc="outer", options=above) as outer:
        outer.panel("the note")
        for line in range(3):
            outer.write(f"line {line}")
        Bar(total=1, desc="kept", options=kept).close()
        for line in range(3, 9):
            outer.write(f"line {line}")
        outer.log("logged")
        outer.update()

    # Assert
    final = _final(fake_ipython)
    assert final[:2] == ["  +5 earlier lines", "| the note"]
    assert final[2].split()[1] == "kept"
    assert final[3:8] == [f"line {line}" for line in range(4, 9)]


@pytest.mark.usefixtures("never_busy")
def test_text_kept_across_sessions_and_dropped_bars_joins_few_blocks(
    fake_ipython: list[FakeHandle],
) -> None:
    # Arrange
    stream = io.StringIO()
    kept = Options(stream=stream, leave=True)

    # Act: sessions keeping no bar, and sessions whose kept bars drop every
    # bar between their text and the text before it
    with engine_lock():
        for session in range(60):
            options = Options(stream=stream, leave=session % 3 == 0)
            with Bar(total=1, desc="s", options=options) as bar:
                bar.write(f"session {session}")
                for index in range(NOTEBOOK_KEPT * (session % 2)):
                    bar.write(f"file {index}")
                    Bar(total=1, desc=f"k{index}", options=kept).close()
                bar.update()
        output = _backends._OUTPUTS[next(iter(_backends._OUTPUTS))]
        held = len(output.kept)

    # Assert: a block per session or per dropped bar would grow the output's
    # items without end, and keeping more would slow down
    assert held <= 2 * NOTEBOOK_KEPT + 2
    final = _final(fake_ipython)
    sessions = [row for row in final if row.startswith("session ")]
    assert sessions == [f"session {session}" for session in range(60)]


def test_a_session_started_after_its_cell_ended_ends_complete(
    fake_ipython: list[FakeHandle], cell_events: FakeEvents
) -> None:
    # Arrange: the cell ends before a thread it started makes its first bar,
    # the hooks listening already, as they do from psnek's import
    stream = io.StringIO()
    cell_events.end_cell()

    # Act
    with engine_lock(), Bar(total=1, options=Options(stream=stream)) as bar:
        for line in range(30):
            bar.write(f"line {line}")
        bar.update()
    settle_idle_engines()

    # Assert
    final = _final(fake_ipython)
    assert [row for row in final if row.startswith("line ")] == [
        f"line {line}" for line in range(30)
    ]


def test_the_cell_end_draws_bars_a_new_engine_started_in_the_cell(
    fake_ipython: list[FakeHandle], cell_events: FakeEvents
) -> None:
    # Arrange: the first session's engine has gone idle, so the next bar
    # starts another, which has not drawn yet
    stream = io.StringIO()
    options = Options(stream=stream)
    with Bar(total=1, desc="first", options=options) as bar:
        bar.update()
    settle_idle_engines()

    # Act
    with engine_lock():
        late = Bar(total=5, desc="started late", options=options)
        late.write("late line")
        cell_events.end_cell()
        complete = fake_ipython[0].updates[-1]["text/plain"]
    late.close()

    # Assert
    assert "started late" in complete
    assert "late line" in complete


@pytest.mark.usefixtures("never_busy")
def test_a_cell_after_an_ended_one_still_trims_its_session_ends(
    fake_ipython: list[FakeHandle], cell_events: FakeEvents
) -> None:
    # Arrange
    shell = sys.modules["IPython"].get_ipython()
    kept = Options(stream=io.StringIO(), leave=True)
    with engine_lock():
        Bar(total=1, desc="first cell", options=kept).close()
    cell_events.end_cell()
    shell.execution_count = 2
    cell_events.start_cell()

    # Act: more bars than an update before the cell's end shows
    with engine_lock():
        for index in range(NOTEBOOK_LIVE_BARS + 1):
            Bar(total=1, desc=f"bar {index}", options=kept).close()

    # Assert: only the first cell ended, so this one's end sends it all later
    assert len(fake_ipython) == 2
    assert "text/plain" not in fake_ipython[1].updates[-1]


_Parent = dict[str, dict[str, object]]


def _cell_request(msg_id: str, msg_type: str = "execute_request") -> _Parent:
    """Return a kernel request's parent message, as ``get_parent`` gives it."""
    return {"header": {"msg_id": msg_id, "msg_type": msg_type}}


_SILENT: _Parent = {**_cell_request("silent-1"), "content": {"silent": True}}


def _on_thread(function: Callable[[], str]) -> str:
    """Return what ``function`` returns on a thread of its own."""
    results: list[str] = []
    thread = threading.Thread(target=lambda: results.append(function()))
    thread.start()
    thread.join()
    return results[0]


def test_requests_running_no_cell_belong_to_the_cell_they_run_in(
    cell_events: FakeEvents,
) -> None:
    # Arrange
    cell_events.end_cell()
    parents = [_cell_request("cell-a")]
    shell = sys.modules["IPython"].get_ipython()
    shell.kernel = types.SimpleNamespace(get_parent=lambda: parents[-1])
    cell_events.start_cell()

    # Act: a widget callback polled inside the running cell, then after it,
    # the first import of psnek in such a callback, and threads, whose
    # parent is whatever shell request came last
    parents.append(_cell_request("comm-1", "comm_msg"))
    polled = notebook_key()
    cell_events.end_cell()
    callback = notebook_key()
    with engine_lock():
        _backends.start_cell()
    callback_ended = cell_ended()
    threads = []
    for other in (_cell_request("complete-1", "complete_request"), _SILENT):
        parents.append(other)
        threads.append(_on_thread(notebook_key))
    parents.append(_cell_request("cell-b"))
    cell_events.start_cell()
    later_ended = cell_ended()
    parents.append(_cell_request("cell-a"))
    task_ended = cell_ended()

    # Assert: an asyncio task keeps its cell's request as its parent, so it
    # stays in its ended cell while a later one runs
    assert polled == "notebook:cell-a"
    assert callback == "notebook:comm-1"
    assert callback_ended
    assert threads == ["notebook:cell-a", "notebook:cell-a"]
    assert not later_ended
    assert task_ended


def test_bars_before_any_cell_runs_key_by_what_the_kernel_knows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange: psnek imported outside a cell, so no cell is known to have run
    parents: list[_Parent] = []

    def get_parent() -> _Parent:
        if not parents:
            raise LookupError
        return parents[-1]

    shell = types.SimpleNamespace(
        kernel=types.SimpleNamespace(get_parent=get_parent), execution_count=4
    )
    module = types.ModuleType("IPython")
    monkeypatch.setattr(module, "get_ipython", lambda: shell, raising=False)
    monkeypatch.setitem(sys.modules, "IPython", module)

    # Act: before the kernel's first request, then on a thread
    first = notebook_key()
    _deferred.run_deferred()
    parents.append(_cell_request("complete-1", "complete_request"))
    thread = _on_thread(notebook_key)

    # Assert
    assert first == "notebook:4"
    assert thread == "notebook:complete-1"


def test_a_cell_run_inside_another_ends_complete_and_leaves_it_running(
    fake_ipython: list[FakeHandle], cell_events: FakeEvents
) -> None:
    # Arrange
    stream = io.StringIO()
    bar = Bar(total=5, desc="open", options=Options(stream=stream))
    for line in range(30):
        bar.write(f"line {line}")

    # Act: as ``%%capture`` or ``run_cell`` runs its body
    with engine_lock():
        engine_for(stream).draw(time.monotonic())
        cell_events.start_cell()
        cell_events.end_cell()
        inner = fake_ipython[0].updates[-1]
        running = not cell_ended()
    bar.close()
    settle_idle_engines()

    # Assert: the outer cell's end, still to come, sends the session's end whole
    assert "line 0" in inner["text/plain"]
    assert running
    assert "text/plain" not in fake_ipython[0].updates[-1]


def test_a_running_cell_keeps_its_output_however_many_cells_run_meanwhile(
    fake_ipython: list[FakeHandle], cell_events: FakeEvents
) -> None:
    # Arrange: the cell's engine has gone idle between its sessions
    cell_events.end_cell()
    parents = [_cell_request("cell-a")]
    shell = sys.modules["IPython"].get_ipython()
    shell.kernel = types.SimpleNamespace(get_parent=lambda: parents[-1])
    cell_events.start_cell()
    options = Options(stream=io.StringIO())
    with Bar(total=1, desc="first", options=options) as bar:
        for line in range(30):
            bar.write(f"line {line}")
        bar.update()
    settle_idle_engines()

    # Act: as subshells run their cells
    for cell in range(_backends.NOTEBOOK_CELLS):
        parents.append(_cell_request(f"other-{cell}"))
        with Bar(total=1, desc="other", options=options) as bar:
            bar.update()
        settle_idle_engines()
    parents.append(_cell_request("cell-a"))
    with engine_lock():
        cell_events.end_cell()

    # Assert
    assert "line 0" in fake_ipython[0].updates[-1]["text/plain"]


@pytest.mark.integration
def test_rerunning_a_cell_in_a_kernel_saves_no_widgets_and_no_growth() -> None:
    # Arrange: progress widgets left open pile up in the saved notebook's
    # widget state on every rerun; psnek's output must save the same each time
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    source = (
        "import psnek\n"
        "for epoch in range(3):\n"
        "    for _ in psnek.track(range(20), desc=f'epoch {epoch}'):\n"
        "        pass\n"
        "    print(f'epoch {epoch} done')\n"
        "with psnek.Bar(total=5, desc='manual') as bar:\n"
        "    bar.update(5)\n"
    )
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(source)])
    client = nbclient.NotebookClient(notebook, timeout=60, kernel_name="python3")
    cell = notebook.cells[0]

    # Act
    saves = []
    with client.setup_kernel():
        for _ in range(5):
            client.execute_cell(cell, 0)
            client.set_widgets_metadata()
            mimes = frozenset(
                mime for output in cell.outputs for mime in output.get("data", {})
            )
            saves.append((len(nbformat.writes(notebook)), len(cell.outputs), mimes))

    # Assert
    sizes, counts, kinds = zip(*saves, strict=True)
    assert "widgets" not in notebook.metadata
    assert set(kinds) == {frozenset({"text/html", "text/plain"})}
    assert len(set(counts)) == 1
    assert max(sizes) - min(sizes) < min(sizes) // 20


@pytest.mark.integration
def test_importing_psnek_in_ipython_listens_for_cells_from_the_one_running() -> None:
    # Act: a cell may end before its first bar starts, as when it starts a
    # thread that makes bars later, so the hooks listen from the import
    printed = {
        case: subprocess.run(
            [
                sys.executable,
                "-c",
                "import os, sys, threading, types\n"
                "events = []\n"
                "register = lambda name, hook: events.append(name)\n"
                "hooks = types.SimpleNamespace(register=register)\n"
                "case = os.environ['CASE']\n"
                "result = None if case == 'between cells' else object()\n"
                "shown = types.SimpleNamespace(exec_result=result)\n"
                "shell = types.SimpleNamespace(\n"
                "    events=hooks, execution_count=7, displayhook=shown\n"
                ")\n"
                "module = types.ModuleType('IPython')\n"
                "module.get_ipython = lambda: shell\n"
                "sys.modules['IPython'] = module\n"
                "if case == 'in a thread':\n"
                "    importer = threading.Thread(target=__import__, args=('psnek',))\n"
                "    importer.start()\n"
                "    importer.join()\n"
                "import psnek._backends\n"
                "print(events, list(psnek._backends._RUNNING.values()))\n",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "CASE": case},
            check=False,
        )
        for case in ("in a cell", "between cells", "in a thread")
    }

    # Assert: only an import inside a running cell knows which cell it is
    hooks = "['pre_run_cell', 'post_run_cell']"
    assert {
        case: (run.returncode, run.stderr) for case, run in printed.items()
    } == dict.fromkeys(printed, (0, ""))
    assert {case: run.stdout for case, run in printed.items()} == {
        "in a cell": f"{hooks} [[7]]\n",
        "between cells": f"{hooks} []\n",
        "in a thread": f"{hooks} []\n",
    }
