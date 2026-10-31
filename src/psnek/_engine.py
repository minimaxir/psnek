"""Output engines: one per stream, driven by a render thread while bars exist."""

from __future__ import annotations

import atexit
import contextvars
import io
import logging
import os
import re
import sys
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import cached_property, lru_cache
from itertools import accumulate, count, repeat
from typing import TYPE_CHECKING, TextIO, TypeVar, cast

from psnek._backends import (
    _OUTPUTS,
    _RUNNING,
    ENGINES,
    NOTEBOOK,
    NOTEBOOK_KEPT,  # noqa: F401 (re-exported)
    TICK,  # noqa: F401
    Backend,
    LogBackend,
    NotebookBackend,
    _hook_cells,
    cell_ended,
    notebook_key,
)
from psnek._deferred import LOCK, defer, describe, run_deferred
from psnek._env import (
    Mode,
    Notify,
    color_depth,
    detect_mode,
    env_float,
    glyphs_for,
    in_notebook,
    notification_protocol,
    once,
    supports_progress_osc,
    terminal_size,
)
from psnek._frame import Frame
from psnek._glyphs import Style
from psnek._redraw import (
    COLUMN_ONE,
    HIDE_CURSOR,
    PROGRESS_CLEAR,
    SHOW_CURSOR,
    SYNC_BEGIN,
    SYNC_END,
    WRAP_OFF,
    WRAP_ON,
    Drawn,
    Placed,
    drawn,
)
from psnek._render import (
    Columns,
    columns_for,
    layout,
    more_bars,
    notice,
    panel_runs,
    printed_runs,
)
from psnek._streams import (
    CAPTURES,
    CAPTURING,
    DISCARD,
    Capture,
    _buffer,  # noqa: F401 (re-exported)
    _restore_streams,
    _tail,  # noqa: F401
    stream_handlers,
    unwrap,
)
from psnek._style import PanelLine, Run, to_ansi
from psnek._text import clean_lines, encodable, ends_at_margin, join_lines, width_bounds

if TYPE_CHECKING:
    from psnek._bar import Bar as Source
    from psnek._bar import Options

BELL = "\a"
LINGER = 1.0
_Row = TypeVar("_Row")
RENDER_ERRORS = (TypeError, ValueError, ArithmeticError, AttributeError, LookupError)
OUTPUT_ERRORS = (OSError, ValueError, TypeError, AttributeError)
_ALMOST_FULL, _PATCH_GAP = 99, 4
_SGR_SEQUENCE = re.compile(r"\x1b\[[0-9;]*m")
_NOTICE_IDS = count(1)
_OSC_STATES = {"stalled": 4, "stuck": 2, "failed": 2, "interrupted": 4}


class _Rendering(threading.local):
    active = False


_RENDERING = _Rendering()


def flush_stdout(target: TextIO, ordered: bool = False) -> None:
    """Flush ``sys.stdout`` before writing to ``target`` (renderers: if ``ordered``)."""
    stdout = sys.stdout
    if stdout is None or stdout is target or (_RENDERING.active and not ordered):
        return
    try:
        stdout.flush()
    except (*OUTPUT_ERRORS, RuntimeError) as error:  # RuntimeError: reentrant
        once(logging.DEBUG, f"could not flush stdout: {describe(error)}")


def _osc(kind: int, percent: int) -> str:
    return f"\x1b]9;4;{kind};{percent}\x1b\\"


def _osc_state(frame: Frame) -> tuple[int, int]:
    kind = _OSC_STATES.get(frame.state, 1)
    fraction = (
        1.0 if frame.fraction is None and frame.state == "done" else frame.fraction
    )
    if fraction is None:
        return 3 if kind == 1 else kind, 0
    percent = round(fraction * 100)
    return kind, percent if frame.full else min(percent, _ALMOST_FULL)


def _notification(protocol: Notify, title: str, body: str) -> str:
    """Return the sequence showing a desktop notification (OSC 99: if unfocused)."""
    title = title.replace(";", ",")
    if protocol == "osc9":
        return f"\x1b]9;{title}: {body}\x1b\\"
    if protocol == "osc777":
        return f"\x1b]777;notify;{title};{body}\x1b\\"
    if protocol != "osc99":
        return BELL
    key = f"i=psnek{next(_NOTICE_IDS)}"
    return (
        f"\x1b]99;{key}:d=0:o=unfocused;{title}\x1b\\\x1b]99;{key}:p=body;{body}\x1b\\"
    )


@lru_cache(maxsize=256)
def _panel_row(line: PanelLine, width: int, style: Style) -> Drawn:
    """Render a panel line once, not on every tick it stays shown."""
    return drawn(panel_runs(line, width, style))


class LiveBackend(Backend):
    """In-place redraw of a multi-line region on an interactive terminal."""

    def __init__(self, stream: TextIO) -> None:
        """Bind to the terminal ``stream`` to draw on."""
        self.style = Style(glyphs=glyphs_for(stream), depth=color_depth(stream))
        self.osc, self.notify = (
            supports_progress_osc(stream),
            notification_protocol(stream),
        )
        self.region: list[Drawn] = []
        self.size = self.measured = self.osc_state = (0, 0)
        self.columns: Columns | None = None
        self.osc_time = self.since = 0.0
        self.closing = PROGRESS_CLEAR
        self.notify_after = (
            env_float("PSNEK_NOTIFY_AFTER", 30.0, 0.0, 86400.0) if self.notify else 0.0
        )
        self.notices: list[str] = []
        self.alerted = False
        self.stuck: weakref.WeakSet[Source] = weakref.WeakSet()

    def _line(self, runs: list[Run]) -> str:
        return to_ansi(runs, self.style.depth)

    def _width(self, engine: Engine) -> int:
        return terminal_size(engine.stream)[0] - 1

    def capacity(self, engine: Engine) -> int:
        """Measure the size the next `draw` lays out; return its height less one."""
        self.measured = terminal_size(engine.stream)
        return max(self.measured[1] - 1, 1)

    def started(self, engine: Engine, source: Source, now: float) -> None:
        """Time the session from its first bar."""
        if len(engine.sources) == 1:
            self.since = now

    def _notify(self, source: Source, frame: Frame) -> None:
        if self.notify is not None and source.notify:
            self.notices.append(_notification(self.notify, *notice(frame)))

    def finished(
        self, engine: Engine, source: Source, frame: Frame, keep: bool
    ) -> None:
        """Commit a kept bar's final line to scrollback; notify of a failure or end."""
        if keep:
            if engine.sources:
                engine.commit()
            width = self._width(engine)
            own, shared = columns_for([frame], width // 3), self.columns
            if shared is None:
                columns = replace(own, state=0)
            else:
                columns = replace(shared, percent=shared.percent or own.percent)
            engine.scrollback += map(
                self._line, layout(frame, width, self.style, columns)
            )
        if self.osc and not engine.sources and frame.state in _OSC_STATES:
            self.closing = _osc(*_osc_state(frame))
        if not source.notify or time.monotonic() - self.since < self.notify_after:
            return
        if frame.state == "failed" and not self.alerted:
            self.alerted = True
            self._notify(source, frame)
        elif frame.state == "done" and not engine.sources:
            self._notify(source, frame)

    def _progress_osc(self, frames: list[Frame], now: float) -> str:
        if not self.osc or not frames:
            return ""
        state = _osc_state(frames[0])
        if state == self.osc_state and now - self.osc_time < 1.0:
            return ""
        self.osc_state, self.osc_time = state, now
        return _osc(*state)

    def draw(self, engine: Engine, frames: list[Frame], now: float) -> None:
        """Redraw the live region in one synchronized write of what changed."""
        size = self.measured
        rows = self._region(engine, frames, size)
        scroll = engine.scrollback[:]
        osc = self._progress_osc(frames, now)
        watched = zip(engine.drawn, frames, strict=True) if self.notify else ()
        for source, frame in watched:  # notify once each time a bar gets stuck
            if frame.state != "stuck":
                self.stuck.discard(source)
            elif source not in self.stuck:
                self.stuck.add(source)
                self._notify(source, frame)
        notices = "".join(self.notices)
        self.notices.clear()
        unchanged = rows == self.region and size == self.size
        if not scroll and not osc and not notices and unchanged:
            return
        parts = [SYNC_BEGIN]
        if not self.region:
            parts += [COLUMN_ONE, HIDE_CURSOR, WRAP_OFF]
        elif size[1] >= self.size[1]:
            parts.append(f"\x1b[{len(self.region)}F")
        if scroll:
            parts += [WRAP_ON, *map(self._over, scroll, repeat(size[0])), WRAP_OFF]
        resized = bool(scroll) or size != self.size
        if resized or len(rows) != len(self.region):
            parts += [f"{self._line(row.runs)}\x1b[K\n" for row in rows]
        else:
            parts += map(self._patch, rows, self.region)
        if resized or len(rows) < len(self.region):
            parts.append("\x1b[J")
        parts += [osc, notices, SYNC_END]
        engine.emit("".join(parts))
        if not engine.broken:
            del engine.scrollback[: len(scroll)]
        self.region, self.size = rows, size

    def _patch(self, row: Drawn, old: Drawn) -> str:
        """Return the writes turning drawn row ``old`` into ``row``, then a newline.

        Unchanged runs are skipped; changed disputed-width runs rewrite the row.
        """
        if row == old:
            return "\n"
        if not (row.placed and row.patchable) or row.disputed != old.disputed:
            return f"{self._line(row.runs)}\x1b[K\n"
        on_screen, unsure = set(old.placed), set(row.disputed)
        parts: list[str] = []
        group: list[Run] = []
        skipped: list[Placed] = []
        cursor = 0

        def move(column: int, runs: list[Placed]) -> None:
            """Bring the cursor to ``column`` over agreed-width ``runs``."""
            nonlocal cursor
            if column - cursor > _PATCH_GAP:
                if group:
                    parts.append(self._line(group))
                    group.clear()
                parts.append(f"\x1b[{column - cursor}C")
            else:
                group.extend((text, ink) for _, text, ink in runs)
            cursor = column

        def reach(column: int) -> None:
            """Bring the cursor to ``column``, redrawing skipped disputed runs."""
            nonlocal cursor
            rest = skipped
            marks = [index for index, run in enumerate(skipped) if run in unsure]
            if marks:
                first, last = marks[0], marks[-1] + 1
                move(skipped[first][0], skipped[:first])
                group.extend((text, ink) for _, text, ink in skipped[first:last])
                rest = skipped[last:]
                cursor = rest[0][0] if rest else column
            move(column, rest)
            skipped.clear()

        ends = [column for column, _, _ in row.placed[1:]] + [row.width]
        for run, end in zip(row.placed, ends, strict=True):
            if run in on_screen:
                skipped.append(run)
                continue
            if skipped:
                reach(run[0])
            group.append(run[1:])
            cursor = end
        if row.width < old.width and skipped:
            reach(row.width)
        if group:
            parts.append(self._line(group))
        return "".join(parts) + ("\x1b[K\n" if row.width < old.width else "\n")

    def _region(
        self, engine: Engine, frames: list[Frame], size: tuple[int, int]
    ) -> list[Drawn]:
        width, budget = size[0] - 1, max(size[1] - 1, 1)
        total, shown = len(engine.sources), frames
        if total > len(frames) or sum(frame.height for frame in frames) > budget:
            heights = accumulate(frame.height for frame in frames)
            shown = frames[: sum(used < budget for used in heights)]
        floor = self.columns or columns_for((), width // 3)
        self.columns = columns_for(shown, width // 3, floor)
        bars = [
            drawn(row)
            for frame in shown
            for row in layout(frame, width, self.style, self.columns)
        ]
        if len(shown) < total:
            bars.append(drawn(more_bars(total - len(shown), width)))
        room = max(budget - len(bars), 0)
        engine.spill(room)
        return engine.with_panel(
            bars, room, lambda line: _panel_row(line, width, self.style)
        )

    def keep_note(self, engine: Engine) -> list[str]:
        """Return the note's lines as drawn, to stay in the transcript."""
        width = self._width(engine)
        return [
            self._line(_panel_row(line, width, self.style).runs) for line in engine.note
        ]

    def text(self, line: PanelLine) -> str:
        """Return panel ``line`` as ANSI text for scrollback, logged ones indented."""
        return self._line(printed_runs(line, self.style.glyphs))

    def _over(self, line: PanelLine, columns: int) -> str:
        """Return scrollback ``line`` written over the row it lands on."""
        text = self.text(line)
        shown = _SGR_SEQUENCE.sub("", text)
        least, most = width_bounds(shown)
        if least == most:
            erase = "" if ends_at_margin(shown, columns) else "\x1b[K"
            return f"{text}{erase}\n"
        most += most // columns
        if most >= columns and most // columns * columns >= least:
            return f"\x1b[J{text}\n"
        return f"{text}\x1b[K\n"

    def restore(self) -> str:
        """Return the sequences that end synchronized output and show the cursor."""
        return SYNC_END + WRAP_ON + SHOW_CURSOR + (PROGRESS_CLEAR if self.osc else "")

    def stop(self, engine: Engine) -> None:
        """Restore the terminal and forget the region."""
        engine.emit(WRAP_ON + SHOW_CURSOR + (self.closing if self.osc else ""))
        self.region, self.osc_state, self.closing = [], (0, 0), PROGRESS_CLEAR
        self.columns, self.alerted = None, False
        self.stuck.clear()
        self.notices.clear()


BACKENDS: dict[Mode, Callable[[TextIO], Backend]] = {
    "live": LiveBackend,
    "log": LogBackend,
    "notebook": NotebookBackend,
    "off": Backend,
}


class Engine:
    """Owns one output stream (or notebook cell): its bars, panel, and thread."""

    def __init__(self, stream: TextIO, mode: Mode, hidden: bool) -> None:
        """Create an idle engine for ``stream`` (never a `Capture`) in ``mode``."""
        self.stream, self.mode, self.hidden = stream, mode, hidden
        self.key = (notebook_key() if mode == NOTEBOOK else id(stream), hidden)
        self.sources: list[Source] = []
        self.drawn: list[Source] = []
        self.top: Source | None = None
        self.links: dict[Source, tuple[Source | None, object]] = {}
        self.logs: deque[PanelLine] = deque()
        self.note: tuple[PanelLine, ...] = ()
        self.scrollback: list[PanelLine] = []
        self.note_version = self.panel_lines = 0
        self.panel_above = self.panel_clear = self.broken = self.ticking = False
        self.idle_since: float | None = None
        self.stop_event = threading.Event()

    @cached_property
    def backend(self) -> Backend:
        """The presenter for this engine's mode, only ever read under `LOCK`."""
        return BACKENDS[self.mode](self.stream)

    def keeps(self, source: Source) -> bool:
        """Return whether ``source`` keeps its final line (default: if on top)."""
        return source.leave if source.leave is not None else source is self.top

    def _capturing(self) -> bool:
        return bool(CAPTURING) and CAPTURING[-1] is self

    def panel_view(self) -> list[PanelLine]:
        """Return up to ``panel_lines``: the note, logged, then partial lines."""
        if not self._capturing():
            return [*self.note, *self.logs][: self.panel_lines]
        partials = [text for capture in CAPTURES.values() for text in capture.preview()]
        return [*self.note, *self.logs, *clean_lines(partials)][: self.panel_lines]

    def with_panel(
        self, bars: list[_Row], rows: int, render: Callable[[PanelLine], _Row]
    ) -> list[_Row]:
        """Place up to ``rows`` panel lines, drawn by ``render``, beside ``bars``."""
        panel = [render(line) for line in self.panel_view()[:rows]]
        return panel + bars if self.panel_above else bars + panel

    def take_scrollback(self) -> list[PanelLine]:
        """Remove and return the lines waiting to be printed above the live region."""
        lines, self.scrollback = self.scrollback, []
        return lines

    def spill(self, rows: int) -> None:
        """Move logged lines that do not fit in ``rows`` panel rows to scrollback."""
        reserved = len(self.note)
        if self._capturing():
            reserved += sum(len(capture.pending) for capture in CAPTURES.values())
        keep = max(min(rows, self.panel_lines) - reserved, 0)
        while len(self.logs) > keep:
            self.scrollback.append(self.logs.popleft())

    def commit(self) -> None:
        """Move logged panel lines into scrollback, keeping output in order."""
        self.scrollback += self.logs
        self.logs.clear()

    def _write(self, text: str) -> None:
        try:
            self.stream.write(text)
        except UnicodeEncodeError as error:
            self.stream.write(encodable(text, error.encoding))
        if (flush := getattr(self.stream, "flush", None)) is not None:
            flush()

    def emit(self, text: str) -> None:
        """Write and flush ``text``, never raising; a failure disables live drawing."""
        if not text:
            return
        try:
            self._write(text)
        except Exception as error:  # any stream failure disables live output
            if self.broken:
                message = f"psnek could not write after a failure: {describe(error)}"
                once(logging.DEBUG, message)
            elif self.mode == "live":
                self.fail(error)
            else:
                once(logging.WARNING, f"psnek dropped output: {describe(error)}")

    def emit_lines(self, lines: list[str]) -> None:
        """Print sanitized ``lines`` directly."""
        if lines:
            self.emit(join_lines(lines))

    def fail(self, error: Exception) -> None:
        """Stop drawing after ``error``, restore the terminal, and print the panel."""
        if self.broken:
            return
        self.broken = True
        defer(
            logging.ERROR, f"psnek output disabled after a failure: {describe(error)}"
        )
        pending = list(map(self.backend.text, [*self.take_scrollback(), *self.logs]))
        self.logs.clear()
        restore = self.backend.restore()
        if restore or pending:
            self.emit(restore + ("\n" + join_lines(pending) if pending else ""))
        _release_capture(self)

    def _owner(self) -> Engine | None:
        if self.sources and not self.hidden and not self.broken:
            return self
        keys = ((id(self.stream), False), (notebook_key(), False))
        engines = (ENGINES.get(key) for key in keys)
        visible = next((found for found in engines if found and found.sources), None)
        visible = visible or _same_terminal(self.stream)
        return visible if visible is not None and not visible.broken else None

    def log(self, lines: Sequence[PanelLine]) -> None:
        """Append sanitized ``lines`` to the live panel, or print them without one."""
        with LOCK:
            if (owner := self._live_owner(lines)) is not None:
                owner.logs += lines
                owner.spill(owner.panel_lines)

    def write(self, lines: list[str]) -> None:
        """Print raw ``lines`` permanently above the live region, after logged ones."""
        clean = clean_lines(lines)
        with LOCK:
            if (owner := self._live_owner(clean)) is not None:
                owner.commit()
                owner.scrollback += clean

    def _live_owner(self, clean: Sequence[PanelLine]) -> Engine | None:
        """Return the engine showing a live or notebook panel, else print ``clean``."""
        owner = self._owner()
        if owner is not None and owner.mode in ("live", NOTEBOOK):
            return owner
        target = owner or self
        target.emit_lines(list(map(target.backend.text, clean)))
        return None

    def set_panel(self, layout: Callable[[int], list[PanelLine]]) -> None:
        """Replace the panel's note with ``layout(panel_lines)``, run unlocked."""
        with LOCK:
            if (owner := self._owner()) is None:
                return
            limit = owner.panel_lines
        note = tuple(layout(limit))
        with LOCK:
            if not owner.sources:
                return
            owner.commit()
            if note != owner.note:
                owner.note = note
                owner.note_version += 1

    def draw(self, now: float) -> None:
        """Sample the shown bars at ``now``, tally the rest, and present a frame."""
        try:
            limit = self.backend.capacity(self)
            self.drawn = drawn = self.sources[:limit]
            parents = {self.links[bar][0] for bar in drawn} if len(drawn) > 1 else ()
            frames = [
                source._sample(now, branches, source in parents)
                for source, branches in zip(drawn, self._branches(), strict=True)
            ]
            for source in self.sources[len(drawn) :]:
                source._tally(now)
            if self._capturing():
                for capture in CAPTURES.values():
                    self.logs += clean_lines(capture.reap())
            if not self.broken:
                self.backend.draw(self, frames, now)
            elif self.scrollback:
                self.emit(join_lines(map(self.backend.text, self.take_scrollback())))
        except RENDER_ERRORS as error:
            self.fail(error)

    def _attach(self, source: Source, owner: object) -> None:
        """Insert ``source`` as a child of ``owner``'s newest bar, else of the top."""
        sources = self.sources
        owned = (bar for bar in reversed(sources) if self.links[bar][1] is owner)
        parent = next(owned, self.top if sources else None)
        index = len(sources)
        if parent is not None:
            index = sources.index(parent) + 1
            while index < len(sources) and self._descends(sources[index], parent):
                index += 1
        sources.insert(index, source)
        self.links[source] = (parent, owner)

    def _descends(self, source: Source, ancestor: Source) -> bool:
        parent = self.links[source][0]
        while parent is not None and parent is not ancestor:
            parent = self.links[parent][0]
        return parent is ancestor

    def _detach(self, source: Source) -> None:
        self.sources.remove(source)
        parent, _ = self.links.pop(source)
        for bar, (up, owner) in self.links.items():
            if up is source:
                self.links[bar] = (parent, owner)

    def _branches(self) -> list[tuple[bool, ...]]:
        """Return the tree position (see `Frame.branches`) of each drawn bar."""
        later: dict[Source, bool] = {}
        seen: set[Source | None] = set()
        for bar in reversed(self.sources):
            parent = self.links[bar][0]
            later[bar] = parent in seen
            seen.add(parent)
        positions = []
        for bar in self.drawn:
            flags, node = [], bar
            while (parent := self.links[node][0]) is not None:
                flags.append(later[node])
                node = parent
            positions.append(tuple(reversed(flags)))
        return positions

    def add(self, source: Source, options: Options, now: float, owner: object) -> None:
        """Register a bar, starting a session and thread if idle, under `LOCK`."""
        if first := not self.sources:
            self.top, self.broken, self.idle_since = source, False, None
            self.panel_lines, self.panel_clear = (
                options.panel_lines,
                options.panel_clear,
            )
            self.panel_above = options.panel_position == "above"
        self._attach(source, owner)
        ENGINES.pop(self.key, None)
        ENGINES[self.key] = self
        added = False
        try:
            if first and options.capture and self.mode == "live":
                _acquire_capture(self)
            backend = self.backend
            if first and isinstance(backend, NotebookBackend):
                backend.output.hooked = _hook_cells() and not cell_ended()
            if not self.ticking:
                self.stop_event = threading.Event()
                threading.Thread(
                    target=contextvars.copy_context().run,
                    args=(self._run, self.stop_event, backend.interval),
                    name="psnek-render",
                    daemon=True,
                ).start()
                self.ticking = True
            backend.started(self, source, now)
            added = True
        finally:
            if not added:
                self._forget(source)

    def _forget(self, source: Source) -> None:
        self._detach(source)
        if self.sources:
            return
        _release_capture(self)
        self.backend.stop(self)
        self.top = None
        self._go_idle()

    def _go_idle(self) -> None:
        self.stop_event.set()
        self.ticking = False
        if ENGINES.get(self.key) is self:
            del ENGINES[self.key]

    def remove(self, source: Source, final: Frame) -> None:
        """Unregister bar ``source``, presenting its ``final`` frame, under `LOCK`."""
        if source not in self.sources:
            return
        kept = self.keeps(source)
        try:
            if len(self.sources) == 1:
                _release_capture(self)
        except RENDER_ERRORS as error:
            self.fail(error)
        finally:
            self._detach(source)
            try:
                self.backend.finished(self, source, final, kept)
            except RENDER_ERRORS as error:
                self.fail(error)
            finally:
                if source is self.top:
                    self.top = self.sources[0] if self.sources else None
                if not self.sources:
                    self._end_session(final.height if kept else 0)

    def _end_session(self, kept_rows: int) -> None:
        """Keep the panel (its note unless cleared) above the kept rows, then close."""
        keep = self.note and not self.panel_clear
        lines = [*(self.backend.keep_note(self) if keep else ()), *self.logs]
        rows = kept_rows if self.panel_above and self.mode == "live" else 0
        at = max(len(self.scrollback) - rows, 0)
        self.scrollback[at:at] = lines
        self.logs.clear()
        self.note = ()
        backend = self.backend
        if not self.ticking:
            self.close_idle()
        elif not (isinstance(backend, NotebookBackend) and backend.busy):
            self._close_session()

    def _close_session(self) -> None:
        """Draw a session's end and restore the stream, even if interrupted."""
        drawn = False
        try:
            self.draw(time.monotonic())
            drawn = True
        finally:
            try:
                if not drawn:
                    self.emit(self.backend.restore())
                self.backend.stop(self)
            finally:
                self.idle_since = time.monotonic()

    def close_idle(self) -> None:
        """End an open session of an engine without bars, then go idle."""
        try:
            if self.idle_since is None:
                self._close_session()
        finally:
            self._go_idle()

    def _run(self, stop: threading.Event, interval: float) -> None:
        _RENDERING.active = True
        ordered = self.mode == "log" and _may_share_file(sys.stdout, self.stream)
        try:
            while not stop.wait(interval):
                try:
                    if ordered:
                        flush_stdout(self.stream, ordered=True)
                    with LOCK:
                        if stop.is_set():
                            return
                        now = time.monotonic()
                        if self.sources:
                            self.draw(now)
                        elif self.idle_since is None:
                            self._close_session()
                        elif now - self.idle_since >= LINGER:
                            self._go_idle()
                            return
                except Exception as error:  # the render thread must keep ticking
                    with LOCK:
                        self.fail(error)
                run_deferred()
        finally:
            with LOCK:
                if self.stop_event is stop:
                    self.ticking = False
            try:
                run_deferred()
            finally:
                _RENDERING.active = False


def _acquire_capture(engine: Engine) -> None:
    """Route stdout, stderr, and their logging handlers into ``engine``'s panel."""
    CAPTURING.append(engine)
    if CAPTURES:
        return
    for name in ("stdout", "stderr"):
        current = getattr(sys, name)
        if current is None:
            continue
        original = unwrap(current)
        if _may_share_file(original, engine.stream, strict=True):
            CAPTURES[name] = Capture(original)
            setattr(sys, name, CAPTURES[name])
    proxies = {id(capture.original): capture for capture in CAPTURES.values()}
    for handler in stream_handlers():
        if (proxy := proxies.get(id(unwrap(handler.stream)))) is not None:
            handler.stream = cast("TextIO", proxy)


def _deliver(engine: Engine, original: TextIO, leftovers: list[str]) -> None:
    if not leftovers:
        return
    if not engine.broken:
        engine.log(clean_lines(leftovers))
        return
    try:
        original.write(join_lines(clean_lines(leftovers)))
        original.flush()
    except Exception as error:  # the stream already failed: never raise from it
        defer(logging.DEBUG, f"could not write unterminated output: {describe(error)}")


def _release_capture(engine: Engine) -> None:
    """Stop capturing for ``engine``; restore the streams after the last one."""
    if engine not in CAPTURING:
        return
    try:
        if len(CAPTURING) == 1:
            drained = [(each.original, each.drain()) for each in CAPTURES.values()]
            _restore_streams()
            for original, leftovers in drained:
                _deliver(engine, original, leftovers)
    finally:
        CAPTURING.remove(engine)


def find(options: Options) -> Engine:
    """Return the engine that displays bars with ``options``, creating it, locked."""
    target = sys.stderr if options.stream is None else options.stream
    if target is sys.stderr and sys.stdout is not None and in_notebook():
        target = sys.stdout  # frontends style the kernel's stderr as errors
    hidden = options.disable or target is None
    stream = DISCARD if target is None else unwrap(target)
    engine = ENGINES.get((id(stream), hidden))
    if engine is not None and engine.sources:
        return engine
    mode = engine.mode if engine else "off" if hidden else detect_mode(stream)
    if mode == NOTEBOOK:
        shared = ENGINES.get((notebook_key(), hidden))
    else:
        shared = _same_terminal(stream) if mode == "live" else None
    return shared or engine or Engine(stream, mode, hidden)


def _file_of(stream: TextIO) -> os.stat_result | None:
    try:
        return os.fstat(stream.fileno())
    except io.UnsupportedOperation:
        return None  # an in-memory stream has no file to identify
    except (AttributeError, OSError, ValueError) as error:
        once(logging.DEBUG, f"cannot identify a stream's file: {describe(error)}")
        return None


def _may_share_file(stream: TextIO | None, other: TextIO, strict: bool = False) -> bool:
    """Report whether ``stream`` may reach ``target``'s file (unknown: not strict)."""
    if stream is None or stream is other:
        return stream is not None
    mine, theirs = _file_of(stream), _file_of(other)
    if mine is None or theirs is None:
        return not strict or mine is theirs
    return os.path.samestat(mine, theirs)


def _same_terminal(stream: TextIO) -> Engine | None:
    """Return a live engine drawing on ``stream``'s terminal, sharing one region."""
    candidates = [
        engine
        for engine in ENGINES.values()
        if engine.mode == "live" and engine.sources and not engine.hidden
    ]
    if (device := _file_of(stream) if candidates else None) is None:
        return None
    for engine in candidates:
        other = _file_of(engine.stream)
        if other is not None and os.path.samestat(device, other):
            return engine
    return None


def active_engine() -> Engine | None:
    """Return the working visible engine that most recently started a bar, if any."""
    with LOCK:
        return next(
            (
                engine
                for engine in reversed(ENGINES.values())
                if engine.sources and not engine.hidden and not engine.broken
            ),
            None,
        )


@atexit.register
def _shutdown() -> None:
    with LOCK:
        sources = [source for engine in ENGINES.values() for source in engine.sources]
    for source in reversed(sources):
        try:
            source._abandon()
        except Exception as error:  # exit must finish every bar
            message = f"psnek could not finish a bar at exit: {describe(error)}"
            defer(logging.ERROR, message)
    with LOCK:
        for engine in list(ENGINES.values()):
            if not engine.sources:
                engine.close_idle()
    run_deferred()


def _after_fork_in_child() -> None:
    for capture in CAPTURES.values():
        capture.pending.clear()
        capture.active = False
    _restore_streams()
    CAPTURING.clear()
    for engine in ENGINES.values():
        engine.sources.clear()
        engine.links.clear()
        engine.ticking = False
    ENGINES.clear()
    _OUTPUTS.clear()
    _RUNNING.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
