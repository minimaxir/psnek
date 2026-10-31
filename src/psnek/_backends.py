"""Backends present an engine's frames; all but the live terminal one live here."""

from __future__ import annotations

import importlib
import logging
import math
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import islice
from typing import TYPE_CHECKING, NamedTuple, Protocol, TextIO

from psnek._deferred import LOCK, defer, describe, run_deferred
from psnek._env import env_float, glyphs_for, html_color_depth, ipython_shell, once
from psnek._frame import STOPPED, UNHEALTHY, Frame, State
from psnek._glyphs import ASCII, Style
from psnek._render import (
    Columns,
    columns_for,
    html_block,
    html_rows,
    layout,
    more_bars,
    panel_runs,
    printed_runs,
    summary,
)
from psnek._style import ColorDepth, PanelLine, Run, to_html, to_plain

if TYPE_CHECKING:
    from psnek._bar import Bar as Source
    from psnek._engine import Engine

TICK, LOG_STRETCH = 1 / 12, 50
NOTEBOOK, NOTEBOOK_TICK, NOTEBOOK_CELLS, NOTEBOOK_WIDTH = "notebook", 0.2, 8, 100
NOTEBOOK_KEPT, NOTEBOOK_KEPT_LINES, NOTEBOOK_ROWS = 20, 10_000, 20
NOTEBOOK_LIVE_LINES, NOTEBOOK_LIVE_BARS, NOTEBOOK_RATE = 20, 5, 50_000
ENGINES: dict[tuple[int | str, bool], Engine] = {}


class Backend:
    """Presents frames on one target; this base, for hidden bars, draws nothing."""

    interval = TICK

    def __init__(self, stream: TextIO) -> None:
        """Bind to the engine's output ``stream``."""

    def capacity(self, engine: Engine) -> int | None:
        """Return how many of ``engine``'s bars to draw (None: all), tallying others."""
        return 0

    def draw(self, engine: Engine, frames: list[Frame], now: float) -> None:
        """Present ``frames`` (one per bar in `Engine.drawn`), scrollback, and panel."""

    def started(self, engine: Engine, source: Source, now: float) -> None:
        """React to bar ``source`` joining ``engine`` at monotonic time ``now``."""

    def finished(
        self, engine: Engine, source: Source, frame: Frame, keep: bool
    ) -> None:
        """React to bar ``source`` leaving with its final ``frame``, ``kept`` or not."""

    def keep_note(self, engine: Engine) -> list[str]:
        """Return the transcript lines keeping the note of a session that ended."""
        return []

    def text(self, line: PanelLine) -> str:
        """Return a panel line as text printed outside the panel, here plain."""
        return line if isinstance(line, str) else to_plain(line)

    def restore(self) -> str:
        """Return the text that resets the target to its normal state."""
        return ""

    def stop(self, engine: Engine) -> None:
        """Release the target once ``engine`` has no bars."""


class LogBackend(Backend):
    """Newline-terminated status lines for CI logs, pipes, and AI agents."""

    def __init__(self, stream: TextIO) -> None:
        """Read the reporting interval; lines go to ``stream`` via the engine."""
        self.every = env_float("PSNEK_LOG_INTERVAL", 10.0, 0.1, 86400.0)
        self.logged: weakref.WeakKeyDictionary[Source, tuple[float, State]] = (
            weakref.WeakKeyDictionary()
        )
        self.announced: weakref.WeakSet[Source] = weakref.WeakSet()
        self.note_version = 0

    def capacity(self, engine: Engine) -> None:
        """Sample every bar, since any kept bar may be due a line."""

    @staticmethod
    def _reports(engine: Engine, source: Source) -> bool:
        return source is engine.top or engine.keeps(source)

    def _note(self, engine: Engine) -> list[str]:
        if engine.note_version == self.note_version:
            return []
        self.note_version = engine.note_version
        return list(map(self.text, engine.note))

    def started(self, engine: Engine, source: Source, now: float) -> None:
        """Announce top-level and kept bars."""
        frame = source._sample(now)
        self.logged[source] = (now, frame.state)
        if self._reports(engine, source):
            self.announced.add(source)
            engine.emit_lines([summary(frame)])

    def finished(
        self, engine: Engine, source: Source, frame: Frame, keep: bool
    ) -> None:
        """Print the final summary of reported, failed, and interrupted bars."""
        _, last = self.logged.pop(source, (0.0, frame.state))
        reported = keep or source is engine.top or source in self.announced
        if reported or last in UNHEALTHY or frame.state in STOPPED:
            engine.emit_lines([summary(frame), *self._note(engine)])

    def draw(self, engine: Engine, frames: list[Frame], now: float) -> None:
        """Emit periodic and health-change summaries."""
        lines = []
        for source, frame in zip(engine.drawn, frames, strict=True):
            last, state = self.logged.get(source, (now, frame.state))
            if self._reports(engine, source):
                due = now - last >= max(self.every, frame.elapsed / LOG_STRETCH)
                report = source not in self.announced or due or frame.state != state
                self.announced.add(source)
            else:
                changed = frame.state != state
                report = changed and not UNHEALTHY.isdisjoint((frame.state, state))
            if report:
                recovered = state in UNHEALTHY and frame.state == "running"
                lines.append(summary(frame) + (" (recovered)" if recovered else ""))
                self.logged[source] = (now, frame.state)
        if lines:
            lines += self._note(engine)
        engine.emit_lines(lines)

    def stop(self, engine: Engine) -> None:
        """Forget per-bar cadence state."""
        self.logged.clear()
        self.announced.clear()


class _DisplayHandle(Protocol):
    """The part of ``IPython.display.DisplayHandle`` psnek uses."""

    def update(self, obj: dict[str, str], *, raw: bool) -> None:
        """Replace the displayed output."""
        ...


class _Text(NamedTuple):
    """Lines printed into a cell's output, rendered as HTML and plain text."""

    rows: deque[tuple[str, str]]

    @property
    def html(self) -> str:
        """The lines as one block whose long lines wrap, not widen it."""
        html = "<br>".join(html for html, _ in self.rows)
        return f'<span style="white-space:pre-wrap">{html}</span>'

    @property
    def text(self) -> str:
        """The lines as plain text."""
        return "\n".join(text for _, text in self.rows)


class _Panel(NamedTuple):
    line: PanelLine


_Kept = Frame | tuple[PanelLine, ...] | _Text
_Shown = Frame | _Text | _Panel | list[Run]
_Tail = tuple[list[_Kept], int, int]


def _recent(kept: deque[_Kept]) -> _Tail:
    """Keep every note and the last bars and lines; count those left out."""
    lines, bars = NOTEBOOK_LIVE_LINES, NOTEBOOK_LIVE_BARS
    shown: list[_Kept] = []
    hidden_lines = hidden_bars = 0
    for item in reversed(kept):
        if isinstance(item, Frame):
            if bars:
                shown.append(item)
                bars -= 1
            else:
                hidden_bars += 1
        elif isinstance(item, _Text):
            rows = deque(islice(reversed(item.rows), lines))
            hidden_lines += len(item.rows) - len(rows)
            lines -= len(rows)
            if rows:
                rows.reverse()
                shown.append(_Text(rows))
        else:
            shown.append(item)
    shown.reverse()
    return shown, hidden_lines, hidden_bars


def _join_texts(kept: deque[_Kept], at: int, spare: _Kept | None = None) -> None:
    """Join the text blocks before and at ``at``, unless the later is ``spare``."""
    if (
        0 < at < len(kept)
        and isinstance(before := kept[at - 1], _Text)
        and isinstance(after := kept[at], _Text)
        and after is not spare
    ):
        before.rows.extend(after.rows)
        del kept[at]


@dataclass(slots=True)
class _CellOutput:
    """The display output shared by every session in one notebook cell."""

    handle: _DisplayHandle | None = None
    kept: deque[_Kept] = field(default_factory=deque)
    items: int = 0
    lines: int = 0
    earlier: int = 0
    earlier_lines: int = 0
    columns: Columns = field(default_factory=lambda: columns_for((), 0))
    bundle: dict[str, str] = field(default_factory=dict)
    failed: bool = False
    sent_at: float = -math.inf
    wait: float = 0.0
    hooked: bool = False
    whole: bool = False
    first: _Kept | None = None
    engine: Engine | None = None


_OUTPUTS: dict[object, _CellOutput] = {}
_RUNNING: dict[threading.Thread, list[object]] = {}
_LATEST: deque[object] = deque([None], maxlen=1)


def _request(shell: object) -> tuple[object, bool]:
    """Return the kernel request being handled, and whether it runs a cell."""
    get_parent = getattr(getattr(shell, "kernel", None), "get_parent", None)
    try:
        parent = (get_parent() or {}) if get_parent else {}
    except LookupError:  # ipykernel before its first request
        once(logging.DEBUG, "no kernel request yet: keying output by execution count")
        parent = {}
    header = parent.get("header", {})
    silent = parent.get("content", {}).get("silent")
    runs = header.get("msg_type") == "execute_request" and not silent
    return header.get("msg_id"), runs


def _current_cell() -> object:
    """Name the executing cell by its kernel request (subshells share a count)."""
    shell = ipython_shell()
    request, runs = _request(shell)
    if runs:
        return request
    thread = threading.current_thread()
    if cells := _RUNNING.get(thread):
        return cells[-1]
    if request is None:
        return getattr(shell, "execution_count", None)
    if thread is threading.main_thread() or _LATEST[0] is None:
        return request
    return _LATEST[0]


def notebook_key() -> str:
    """Return the registry key naming the executing notebook cell's own engine."""
    return f"{NOTEBOOK}:{_current_cell()}"


def _running(cell: object) -> bool:
    return any(cell in cells for cells in _RUNNING.values())


def start_cell() -> None:
    """Mark the executing cell running (nested in one on its thread), under `LOCK`."""
    shell = ipython_shell()
    request, runs = _request(shell)
    if request is not None and not runs:
        return
    cell = request if runs else getattr(shell, "execution_count", None)
    _RUNNING.setdefault(threading.current_thread(), []).append(cell)
    _LATEST.append(cell)


def end_cell() -> Engine | None:
    """Mark the executing cell ended; return the engine that last drew its output."""
    cell = _current_cell()
    thread = threading.current_thread()
    if cells := _RUNNING.get(thread):
        cells.pop()
        if not cells:
            del _RUNNING[thread]
    output = _OUTPUTS.get(cell)
    if output is None:
        return None
    output.hooked, output.whole, output.wait = _running(cell), True, 0.0
    return output.engine


def cell_ended() -> bool:
    """Return whether the executing cell is not running, as for a thread's bars."""
    return not _running(_current_cell())


def _cell_output() -> _CellOutput:
    """Return the executing cell's output, forgetting idle ones beyond a few."""
    cell = _current_cell()
    output = _OUTPUTS.pop(cell, None) or _CellOutput()
    if (excess := len(_OUTPUTS) + 1 - NOTEBOOK_CELLS) > 0:
        idle = (
            key
            for key in _OUTPUTS
            if (f"{NOTEBOOK}:{key}", False) not in ENGINES and not _running(key)
        )
        for key in list(islice(idle, excess)):
            del _OUTPUTS[key]
    _OUTPUTS[cell] = output
    return output


@lru_cache(maxsize=256)
def _bar_html(frame: Frame, columns: Columns, style: Style) -> str:
    rows = layout(frame, NOTEBOOK_WIDTH, style, columns)
    return html_rows([to_html(row, style.depth) for row in rows])


@lru_cache(maxsize=256)
def _bar_text(frame: Frame, columns: Columns, style: Style) -> str:
    return "\n".join(map(to_plain, layout(frame, NOTEBOOK_WIDTH, style, columns)))


@lru_cache(maxsize=256)
def _panel_html(line: PanelLine, style: Style) -> str:
    return to_html(panel_runs(line, NOTEBOOK_WIDTH, style), style.depth)


@lru_cache(maxsize=256)
def _panel_text(line: PanelLine, style: Style) -> str:
    return to_plain(panel_runs(line, NOTEBOOK_WIDTH, style))


def _position(kept: deque[_Kept], item: _Kept | None) -> int:
    """Return where ``item`` itself (not an equal one) is in ``kept``; None: the end."""
    if item is None:
        return len(kept)
    return next(index for index, each in enumerate(kept) if each is item)


class NotebookBackend(Backend):
    """One display output per notebook cell, resent at most every `NOTEBOOK_TICK`."""

    interval = NOTEBOOK_TICK

    def __init__(self, stream: TextIO) -> None:
        """Prepare renderers for the executing cell; ``stream`` picks glyphs."""
        self.color = Style(glyphs=glyphs_for(stream), depth=html_color_depth())
        self.plain = Style(glyphs=ASCII, depth=ColorDepth.NONE)
        self.output = _cell_output()

    def _text_row(self, line: PanelLine) -> tuple[str, str]:
        """Render a written line, or a logged one indented to the bar descriptions."""
        runs = printed_runs(line, self.color.glyphs)
        return to_html(runs, self.color.depth), to_plain(runs)

    def _render(self, shown: _Shown, columns: Columns, html: bool) -> str:
        style = self.color if html else self.plain
        if isinstance(shown, Frame):
            return (_bar_html if html else _bar_text)(shown, columns, style)
        if isinstance(shown, _Text):
            return shown.html if html else shown.text
        if isinstance(shown, _Panel):
            return (_panel_html if html else _panel_text)(shown.line, style)
        return to_html(shown, style.depth) if html else to_plain(shown)

    def _keep_text(self, engine: Engine) -> None:
        """Move the lines printed above the bars into the output (or the cell)."""
        lines = engine.take_scrollback()
        if not lines:
            return
        if self.output.failed:
            engine.emit_lines(list(map(self.text, lines)))
        else:
            self._remember(_Text(deque(map(self._text_row, lines))))

    def capacity(self, engine: Engine) -> int:
        """Return `NOTEBOOK_ROWS`."""
        return NOTEBOOK_ROWS

    def started(self, engine: Engine, source: Source, now: float) -> None:
        """Create the cell's display with its first kept bar, after old text."""
        output = self.output
        if len(engine.sources) == 1:
            output.engine = engine
            self._keep_text(engine)
            if output.first is not None:
                _join_texts(output.kept, _position(output.kept, output.first))
            output.first = None
        frame = source._sample(now)
        if output.handle is None and engine.keeps(source):
            self.draw(engine, [frame], now)

    def finished(
        self, engine: Engine, source: Source, frame: Frame, keep: bool
    ) -> None:
        """List kept bars inside the output after the text printed before them."""
        if keep and engine.sources:
            engine.commit()
        self._keep_text(engine)
        if keep:
            self._remember(frame)

    def keep_note(self, engine: Engine) -> list[str]:
        """Keep the note after the session's bars, or before them when drawn above."""
        if not engine.panel_above:
            self._remember(engine.note)
            return []
        output, note = self.output, engine.note
        kept = output.kept
        kept.insert(_position(kept, output.first), note)
        output.items += 1
        output.first = note
        self._evict(spare=note)
        if engine.logs:
            at = _position(kept, note) + 1
            logged = _Text(deque(map(self._text_row, engine.logs)))
            engine.logs.clear()
            after = kept[at] if at < len(kept) else None
            if isinstance(after, _Text):
                after.rows.extendleft(reversed(logged.rows))
            else:
                kept.insert(at, logged)
            self._count_lines(len(logged.rows))
        return []

    def _remember(self, item: _Kept) -> None:
        """Keep a final bar, note, or text, dropping the oldest past the limits."""
        output = self.output
        kept = output.kept
        if isinstance(item, _Text):
            if output.first is not None and isinstance(kept[-1], _Text):
                kept[-1].rows.extend(item.rows)
            else:
                self._append(item)
            self._count_lines(len(item.rows))
            return
        self._append(item)
        output.items += 1
        if isinstance(item, Frame):
            output.columns = columns_for([item], NOTEBOOK_WIDTH // 3, output.columns)
        self._evict()

    def _append(self, item: _Kept) -> None:
        output = self.output
        output.kept.append(item)
        if output.first is None:
            output.first = item

    def _drop(self, index: int) -> None:
        """Remove kept item ``index``, joining the session's text blocks around it."""
        output = self.output
        dropped = output.kept[index]
        del output.kept[index]
        if dropped is output.first:
            output.first = output.kept[index] if index < len(output.kept) else None
        else:
            _join_texts(output.kept, index, output.first)

    def _evict(self, spare: _Kept | None = None) -> None:
        """Drop the oldest bar or note beyond `NOTEBOOK_KEPT`, other than ``spare``."""
        output = self.output
        if output.items <= NOTEBOOK_KEPT:
            return
        oldest = next(
            index
            for index, item in enumerate(output.kept)
            if not isinstance(item, _Text) and item is not spare
        )
        output.earlier += isinstance(output.kept[oldest], Frame)
        output.items -= 1
        self._drop(oldest)

    def _count_lines(self, added: int) -> None:
        output = self.output
        output.lines += added
        excess = output.lines - NOTEBOOK_KEPT_LINES
        if excess <= 0:
            return
        output.lines -= excess
        output.earlier_lines += excess
        index = 0
        while excess:
            while not isinstance(block := output.kept[index], _Text):
                index += 1
            cut = min(excess, len(block.rows))
            for _ in range(cut):
                block.rows.popleft()
            excess -= cut
            if not block.rows:
                self._drop(index)

    def _shown(self, engine: Engine, frames: list[Frame], tail: _Tail) -> list[_Shown]:
        """List the output's rows: the kept ``tail`` (with counts cut) and bars."""
        output = self.output
        kept, hidden_lines, hidden_bars = tail
        shown: list[_Shown] = []
        for item in [*kept, *frames]:
            if isinstance(item, Frame | _Text):
                shown.append(item)
            else:
                shown += map(_Panel, item)
        if lines := output.earlier_lines + hidden_lines:
            shown.insert(0, more_bars(lines, NOTEBOOK_WIDTH, "earlier", "line"))
        if earlier := output.earlier + hidden_bars:
            shown.insert(0, more_bars(earlier, NOTEBOOK_WIDTH, "earlier"))
        if (unshown := len(engine.sources) - len(frames)) > 0:
            shown.append(more_bars(unshown, NOTEBOOK_WIDTH))
        return engine.with_panel(shown, NOTEBOOK_ROWS, _Panel)

    def draw(self, engine: Engine, frames: list[Frame], now: float) -> None:
        """Update the cell's display in place, printing its text there on failure."""
        output = self.output
        self._keep_text(engine)
        if output.failed:
            return
        if engine.sources and time.monotonic() - output.sent_at < output.wait:
            return
        complete = output.whole or not (engine.sources or output.hooked)
        output.whole = False
        recent = (list(output.kept), 0, 0) if complete else _recent(output.kept)
        complete = complete or not (engine.sources or any(recent[1:]))
        columns = columns_for(frames, NOTEBOOK_WIDTH // 3, output.columns)
        output.columns = columns
        shown = self._shown(engine, frames, recent)
        if not shown and output.handle is None:
            return
        markup = "<br>".join(self._render(item, columns, True) for item in shown)
        bundle = {"text/html": html_block(markup) if shown else ""}
        if complete:
            texts = (self._render(item, columns, False) for item in shown)
            bundle["text/plain"] = "\n".join(texts)
        if bundle == output.bundle:
            return
        try:
            if output.handle is None:
                display = importlib.import_module("IPython.display").display
                output.handle = display(bundle, raw=True, display_id=True)
            else:
                output.handle.update(bundle, raw=True)
        except Exception as error:  # the frontend's failure: stop only this cell
            output.failed = True
            defer(logging.ERROR, f"psnek notebook display disabled: {describe(error)}")
            texts = (item.text for item in output.kept if isinstance(item, _Text))
            engine.emit_lines(list(texts))
            return
        output.bundle = bundle
        output.sent_at = time.monotonic()
        pace = sum(len(data.encode()) for data in bundle.values()) / NOTEBOOK_RATE
        output.wait = pace if not complete and pace > NOTEBOOK_TICK else 0.0

    @property
    def busy(self) -> bool:
        """Whether the display was updated less than a tick ago."""
        return time.monotonic() - self.output.sent_at < NOTEBOOK_TICK


def _finish_cell_outputs(result: object = None) -> None:
    """Draw the cell's output complete as its ``post_run_cell`` hook runs."""
    with LOCK:
        last = end_cell()
        if last is not None:
            last.draw(time.monotonic())
    run_deferred()


def _start_cell(info: object = None) -> None:
    with LOCK:
        start_cell()


def _hook_cells() -> bool:
    """Register the cell hooks with IPython once; return whether it reports cells."""
    events = getattr(ipython_shell(), "events", None)
    if events is None:
        return False
    hooks = getattr(events, "callbacks", {}).get("post_run_cell", ())
    if _finish_cell_outputs not in hooks:
        events.register("pre_run_cell", _start_cell)
        events.register("post_run_cell", _finish_cell_outputs)
    return True


# A cell that ends before its first bar starts, as one starting a thread that
# makes bars later, is known as ended only if the hooks already listen. The
# cell importing psnek on the main thread has started; elsewhere, as in a
# thread or a task between cells, which cell runs is unknown.
if (
    _hook_cells()
    and threading.current_thread() is threading.main_thread()
    and getattr(getattr(ipython_shell(), "displayhook", None), "exec_result", None)
    is not None
):
    _start_cell()
