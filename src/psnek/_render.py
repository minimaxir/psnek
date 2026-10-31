"""Pure rendering: bar snapshots in, styled text runs out, with no clock or I/O."""

import logging
import re
import time
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import chain, islice
from typing import NamedTuple

from psnek._frame import ENDED, STOPPED, UNHEALTHY, Frame
from psnek._glyphs import INDENT, STEPS, Glyphs, Style, tall_head
from psnek._style import (
    BOLD,
    DEFAULT_PALETTE,
    FAINT,
    PLAIN,
    ColorDepth,
    Ink,
    Palette,
    PanelLine,
    Run,
    blend,
    gradient_at,
    to_ansi,
    to_html,
    to_plain,
    to_xterm256,
)
from psnek._text import (
    EMOJI_PRESENTATION,
    clean_lines,
    display_width,
    format_age,
    format_duration,
    format_number,
    format_rate,
    pad,
    sanitize,
    truncate,
    wrap,
)

_HALF, _SPINNER_STEP = STEPS // 2, 0.09
_SHIMMER_RADIUS, _SHIMMER_SPEED, _SHIMMER_STEPS, _SHIMMER_STRENGTH = 2.5, 12.0, 4, 0.45
_COMET_LENGTH, _COMET_MIN_PERIOD, _COMET_CELLS_PER_SECOND = 10, 1.5, 6
_MIN_BAR, _MAX_BAR, _MIN_LABEL, _GAP = 10, 40, 4, 2
_PERCENT_WIDTH, _COUNT_WIDTH, _TIMING_WIDTH, _RATE_WIDTH = 6, 9, 11, 9
_RAW_LIMIT, _NOTICE_TITLE, _NOTICE_BODY = 4096, 60, 200
_MIN_STATUS, _STATUS_RESERVE, _STATUS_MEASURE, _BAR_BEFORE_STATUS = 8, 24, 99, 20
_BLANKS = " \u2800"
_NUMBER = re.compile(
    r"-?\d+(?:[.:]\d+)*"  # digits, decimals and clock times
    r"(?:(?:[kMGTPEZY]|[KMGTPEZY]i)(?![^\W\d_]))?"  # a prefix not starting a unit
    r"%?"
)
_UNSCALED_LIMIT, _ALMOST_DONE = 10**7, 0.999
_HEALTH_SLOT = len("stalled 59m59s")
_STATE_SLOT = _HEALTH_SLOT + _GAP
_STATE, _STATUS = 4, 5
_DROP_ORDER = (3, 2, 1, 0, _STATE)
_SOLID = '<span style="display:inline-block;vertical-align:top;line-height:1">'
PRE_CSS = (
    "margin:0;line-height:1.25;white-space:pre;overflow-x:auto;max-width:100%;"
    "font-family:var(--jp-code-font-family,ui-monospace,Menlo,Consolas,"
    "'DejaVu Sans Mono','Noto Sans Mono','Noto Sans Symbols 2',monospace)"
)


@dataclass(frozen=True, slots=True)
class Columns:
    """Field widths shared by stacked bars so that their fields line up.

    Attributes:
        label: Description width.
        percent: Percentage width; 0 when no bar in the stack has a total.
        count: Count field width.
        timing: Elapsed/ETA field width.
        state: State label width, with room for an idle time; 0 when none is shown.
        rate: Rate field width.
        status: Widest status text.
        bare: Widest state label without a cause, the least ``state`` narrows to.
    """

    label: int = 0
    percent: int = _PERCENT_WIDTH
    count: int = _COUNT_WIDTH
    timing: int = _TIMING_WIDTH
    state: int = 0
    rate: int = _RATE_WIDTH
    status: int = 0
    bare: int = 0


_NARROWEST = Columns(percent=0)


def columns_for(
    frames: Iterable[Frame], label_cap: int, floor: Columns = _NARROWEST
) -> Columns:
    """Measure the widest value of each field across a stack of bars.

    Args:
        frames: The bars drawn together.
        label_cap: Maximum description width (at least four cells).
        floor: Smallest widths to return, so columns only grow while shown.

    Returns:
        Widths that fit every frame.

    Example:
        >>> frames = [Frame(n=5, total=100, desc="a"), Frame(desc="ab")]
        >>> columns = columns_for(frames, 20)
        >>> columns.label, columns.percent, columns.count, columns.state
        (2, 6, 9, 0)
    """
    label, percent, count, timing = (
        floor.label,
        floor.percent,
        floor.count,
        floor.timing,
    )
    state, rate, status, bare = floor.state, floor.rate, floor.status, floor.bare
    label_cap = max(label_cap, _MIN_LABEL)
    for frame in frames:
        if frame.total is not None:
            percent = _PERCENT_WIDTH
        texts = _texts(frame, label_cap)
        label = max(label, texts.label)
        count = max(count, display_width(texts.progress))
        timing = max(timing, len(texts.timing))
        shown = _HEALTH_SLOT if frame.state in UNHEALTHY else display_width(texts.state)
        state = max(state, shown)
        caused = frame.cause and frame.state in STOPPED
        bare = max(bare, len(frame.state) if caused else shown)
        rate = max(rate, display_width(texts.rate))
        status = max(status, _status_width(frame))
    return Columns(label, percent, count, timing, state, rate, status, bare)


class _Texts(NamedTuple):
    """A frame's description lines and fields; ``label`` is their width with indent."""

    desc: tuple[str, ...]
    label: int
    progress: str
    timing: str
    state: str
    rate: str


@lru_cache(maxsize=256)
def _texts(frame: Frame, label_cap: int) -> _Texts:
    """Format ``frame``'s text once for measuring its columns and drawing it."""
    unit, indent = _visible(frame.unit), INDENT * len(frame.branches)
    cap = max(label_cap - indent, _MIN_LABEL)
    if frame.height == 1:
        desc: tuple[str, ...] = (truncate(_visible(frame.desc), cap),)
    else:
        desc = wrap(_visible(frame.desc), cap, frame.height)
    return _Texts(
        desc,
        indent + max(map(display_width, desc), default=0),
        _count_text(frame, unit),
        _timing_text(frame),
        _state_text(frame),
        format_rate(frame.rate, unit, frame.scale),
    )


def _visible(text: str) -> str:
    return sanitize(text[:_RAW_LIMIT])


def _state_color(frame: Frame, palette: Palette) -> int | None:
    if frame.state == "stalled":
        return palette.stalled
    if frame.state in ("stuck", "failed"):
        return palette.failed
    if frame.state == "interrupted":
        return palette.interrupted
    stops = _stops(frame, palette)
    return gradient_at(stops, frame.fraction or 0.0) if stops else None


def _stops(frame: Frame, palette: Palette) -> tuple[int, ...]:
    """Return the frame's own gradient stops, the palette's, or none to go uncolored."""
    if isinstance(frame.gradient, tuple):
        return frame.gradient
    return palette.gradient if frame.gradient else ()


def _tree(frame: Frame, style: Style, below: bool = False) -> list[Run]:
    if not (below or frame.branches):
        return []
    guide, blank, branch, last = style.glyphs.tree
    flags = frame.branches if below else frame.branches[:-1]
    guides = "".join(guide if flag else blank for flag in flags)
    if below:
        end = guide if frame.children else style.glyphs.partials[0] + " "
    else:
        end = branch if frame.branches[-1] else last
    return [(guides + end, FAINT)]


def _spinner(frame: Frame, style: Style) -> Run:
    glyphs = style.glyphs
    if frame.state == "done":
        stops = _stops(frame, style.palette)
        return glyphs.done, Ink(stops[-1] if stops else None, bold=True)
    color = _state_color(frame, style.palette)
    if frame.state in STOPPED:
        return getattr(glyphs, frame.state), Ink(color, bold=True)
    index = int(frame.phase % (_SPINNER_STEP * len(glyphs.spinner)) / _SPINNER_STEP)
    return glyphs.spinner[index], Ink(color)


def shimmer(level: float, palette: Palette, color: int) -> int:
    """Brighten ``color`` toward ``palette.shimmer`` by ``level`` in quantized steps."""
    steps = round(min(max(level, 0.0), 1.0) * _SHIMMER_STEPS) / _SHIMMER_STEPS
    return blend(color, palette.shimmer, steps * _SHIMMER_STRENGTH) if steps else color


@lru_cache(maxsize=256)
def _gradient(
    stops: tuple[int, ...], cells: int
) -> tuple[tuple[int, ...], tuple[Ink, ...]]:
    last = max(cells - 1, 1)
    colors = tuple(gradient_at(stops, index / last) for index in range(cells))
    return colors, tuple(map(Ink, colors))


@lru_cache(maxsize=16)
def _path(height: int) -> tuple[tuple[int, int], ...]:
    """Return each step's row and cell fill step: up the left dots, down the right."""
    steps = range(_HALF * height)
    climb = [(height - 1 - step // _HALF, step % _HALF) for step in steps]
    return (*climb, *((step // _HALF, _HALF + step % _HALF) for step in steps))


@lru_cache(maxsize=256)
def _levels(height: int, part: int) -> tuple[int, ...]:
    """Return the fill steps reached in each row of a column ``part`` steps along."""
    levels = [0] * height
    for row, _ in _path(height)[:part]:
        levels[row] += 1
    return tuple(levels)


def _floor(style: Style, row: int, height: int) -> str:
    """Return the unfilled glyph of ``row``: the track, or an empty fill step above."""
    return style.glyphs.track if row == height - 1 else style.glyphs.partials[0]


def _determinate_cells(
    frame: Frame, fraction: float, cells: int, style: Style
) -> list[list[Run]]:
    palette, height, partials = style.palette, frame.height, style.glyphs.partials
    steps = STEPS * height
    total = round(fraction * cells * steps)
    full, part = divmod(total if frame.full else min(total, cells * steps - 1), steps)
    filled = min(full + (part > 0), cells)
    stops = _stops(frame, palette)
    if stops and frame.state in ("running", "done"):
        colors, inks = _gradient(stops, cells)
    else:
        colors, inks = (), (Ink(_state_color(frame, palette)),) * cells
    lit: dict[int, Ink] = {}
    if colors and frame.state == "running":
        period = (cells + 2 * _SHIMMER_RADIUS + 1) / _SHIMMER_SPEED
        center = frame.phase % period * _SHIMMER_SPEED - _SHIMMER_RADIUS
        start = max(int(center - _SHIMMER_RADIUS), 0)
        for index in range(start, min(int(center + _SHIMMER_RADIUS) + 1, filled)):
            if (distance := abs(index - center)) < _SHIMMER_RADIUS:
                level = 1 - distance / _SHIMMER_RADIUS
                lit[index] = Ink(shimmer(level, palette, colors[index]))
    shown_inks, unfilled = inks[:filled], cells - filled
    rows = []
    for row, level in enumerate(_levels(height, part)):
        floor = _floor(style, row, height)
        shown = [partials[STEPS]] * full + [partials[level]] * (filled - full)
        runs: list[Run] = list(zip(shown, shown_inks, strict=True))
        for index, ink in lit.items():
            runs[index] = (shown[index], ink)
        if filled > full and not level:
            runs[full] = (floor, FAINT)
        if unfilled:
            runs.append((floor * unfilled, FAINT))
        rows.append(runs)
    return rows


def _comet_cells(frame: Frame, cells: int, style: Style) -> list[list[Run]]:
    glyphs, height = style.glyphs, frame.height
    steps = STEPS * height
    moods = {"stalled": glyphs.stalled_heads, "stuck": glyphs.stuck_heads}
    pair = moods.get(frame.state) or glyphs.comet_heads
    heads = None if pair is None else [tall_head(head, height) for head in pair]
    reach = display_width(heads[0][0]) if heads else 0
    if 2 * reach > cells:  # a face with less room to travel than its width stalls
        heads, reach = None, 0
    positions = (cells - reach) * steps
    length = min(_COMET_LENGTH * height, positions)
    period = max(_COMET_MIN_PERIOD, cells / _COMET_CELLS_PER_SECOND)
    progress = (frame.phase % (2 * period)) / period
    forward = progress < 1
    bounce = progress if forward else 2 - progress
    start = round(bounce * bounce * (3 - 2 * bounce) * (positions - length))
    start += 0 if forward else reach * steps
    masks = [[0] * cells for _ in range(height)]
    path, bits = _path(height), glyphs.dot_bits
    for position in range(start, start + length):
        cell, step = divmod(position, steps)
        row, dot = path[step]
        masks[row][cell] |= bits[dot] if bits else 1
    color = _state_color(frame, style.palette)
    first_cell, last_cell = start // steps, (start + length - 1) // steps
    head, body = _comet_head(frame, style, color), Ink(color)
    rows = []
    for row, row_masks in enumerate(masks):
        floor = _floor(style, row, height)
        runs: list[Run] = [
            (chr(0x2800 | mask) if bits else glyphs.partials[STEPS], body)
            if mask
            else (floor, FAINT)
            for mask in row_masks
        ]
        if heads:
            at = last_cell + 1 if forward else first_cell - reach
            # Bold Braille can fall back to a font that outlines the unraised dots.
            runs[at : at + reach] = [(heads[not forward][row], Ink(head.color))]
        elif first_cell != last_cell:
            lead = last_cell if forward else first_cell
            if row_masks[lead]:
                uncolored = style.depth is ColorDepth.NONE
                runs[lead] = (
                    glyphs.partials[STEPS] if uncolored else runs[lead][0],
                    head,
                )
        rows.append(runs)
    return rows


def _comet_head(frame: Frame, style: Style, color: int | None) -> Ink:
    if color is None or frame.state != "running":
        return Ink(color, bold=True)
    shown = to_xterm256 if style.depth is ColorDepth.EIGHT_BIT else int
    for head in (shimmer(1.0, style.palette, color), style.palette.shimmer):
        if shown(head) != shown(color):
            return Ink(head)
    return Ink(color, bold=True)


def _bar_cells(frame: Frame, cells: int, style: Style) -> list[list[Run]]:
    fraction = (
        1.0 if frame.fraction is None and frame.state == "done" else frame.fraction
    )
    if fraction is not None:
        return _determinate_cells(frame, fraction, cells, style)
    if frame.state not in STOPPED:
        return _comet_cells(frame, cells, style)
    ink, height = Ink(_state_color(frame, style.palette)), frame.height
    return [
        [(_floor(style, row, height) * cells, ink if row == height - 1 else FAINT)]
        for row in range(height)
    ]


def _count_text(frame: Frame, unit: str) -> str:
    """Return ``n/total``; unscaled counts of eight or more digits use SI."""
    scale = frame.scale
    if scale == "none" and max(abs(frame.n), frame.total or 0) >= _UNSCALED_LIMIT:
        scale = "si"
    suffix = "" if frame.scale == "none" else unit
    done = format_number(frame.n, scale) + suffix
    if frame.total is None:
        return done
    total = format_number(frame.total, scale) + suffix
    return f"{done:>{len(total)}}/{total}"


def _timing_text(frame: Frame) -> str:
    elapsed = format_duration(frame.elapsed)
    if frame.state in ENDED:
        return elapsed
    mark = "~" if frame.approximate and frame.eta is not None else ""
    return f"{elapsed}<{mark}{format_duration(frame.eta)}"


def _state_text(frame: Frame) -> str:
    if frame.state == "running":
        return ""
    if frame.state in UNHEALTHY:
        return f"{frame.state} {format_age(frame.idle)}"
    if frame.cause and frame.state in STOPPED:
        return f"{frame.state} {_visible(frame.cause)}"
    return frame.state


def _percent_text(frame: Frame) -> str:
    if (fraction := frame.fraction) is None:
        return " " * _PERCENT_WIDTH
    return f"{(fraction if frame.full else min(fraction, _ALMOST_DONE)) * 100:5.1f}%"


_Pair = tuple[str, str]
_SPACER: Run = ("  ", PLAIN)


def _status_pairs(frame: Frame) -> Iterator[_Pair]:
    """Sanitize the status pairs lazily, so a bar's many pairs cost per shown pair."""
    return ((_visible(key), _visible(value)) for key, value in frame.status)


def _styled(key: str, text: str, tail: str = "") -> tuple[Run, ...]:
    """Style a status pair's ``text`` as in the panel: a bold key, faint ``=``."""
    size = len(key)
    parts = (text[:size], text[size : size + 1], text[size + 1 :] + tail)
    return tuple(run for run in zip(parts, (BOLD, FAINT, PLAIN), strict=True) if run[0])


@lru_cache(maxsize=1024)
def _pair(key: str, value: str) -> tuple[int, tuple[Run, ...]]:
    """Measure and style a status pair, once while its value holds."""
    text = f"{key}={value}"
    return display_width(text), _styled(key, text)


def _cut_pair(key: str, value: str, room: int) -> tuple[Run, ...]:
    """Style a status pair shortened to ``room`` cells (ending in ``...``) if wider."""
    text = f"{key}={value}"
    cut = truncate(text, room)
    return _styled(key, text) if cut == text else _styled(key, cut[:-3], "...")


def _status_width(frame: Frame) -> int:
    used = -_GAP
    for pair in _status_pairs(frame):
        used += _GAP + _pair(*pair)[0]
        if used >= _STATUS_MEASURE:
            return _STATUS_MEASURE
    return max(used, 0)


def _fit_status(pairs: Iterator[_Pair], room: int) -> tuple[Run, ...]:
    """Lay out the status pairs fitting in ``room`` cells, shortening only the first."""
    if (first := next(pairs, None)) is None:
        return ()
    shown: list[tuple[Run, ...]] = []
    used = -_GAP
    for pair in chain((first,), pairs):
        width, styled = _pair(*pair)
        used += _GAP + width
        if used > room:
            break
        shown.append(styled)
    if not shown:
        return _cut_pair(*first, room) if room >= _MIN_STATUS else ()
    return (*shown[0], *chain.from_iterable((_SPACER, *runs) for runs in shown[1:]))


@lru_cache(maxsize=1024)
def _quantity(text: str, ink: Ink = PLAIN) -> tuple[Run, ...]:
    """Split ``text`` into numbers (with ``%`` or a bare prefix) and faint units."""
    runs: list[Run] = []
    position = 0
    for match in _NUMBER.finditer(text):
        if between := text[position : match.start()]:
            runs.append((between, PLAIN if between.isspace() else FAINT))
        runs.append((match.group(), ink))
        position = match.end()
    if position < len(text):
        runs.append((text[position:], FAINT))
    return tuple(runs)


def _fit_causes(
    frame: Frame, texts: _Texts, columns: Columns, room: int
) -> tuple[_Texts, Columns]:
    """Cut causes and narrow the state column to leave the bar its least width."""
    least = room - _MIN_BAR - _GAP
    floor = max(least, columns.bare)
    if columns.state > floor:
        columns = replace(columns, state=floor)
    cap = floor if frame.height == 1 else least
    stopped = frame.cause and frame.state in STOPPED
    if not stopped or display_width(texts.state) <= cap:
        return texts, columns
    cut = truncate(texts.state, cap)
    short = display_width(cut) < len(frame.state) + len(" xx...")
    return texts._replace(state=frame.state if short else cut), columns


def _fields(frame: Frame, texts: _Texts, columns: Columns) -> list[Sequence[Run]]:
    """Return the fields in display order, with an empty slot for the status."""
    state = pad(texts.state, columns.state, left=False)
    return [
        _quantity(_percent_text(frame), BOLD) if columns.percent else (),
        _quantity(pad(texts.progress, columns.count)),
        _quantity(pad(texts.timing, columns.timing)),
        _quantity(pad(texts.rate, columns.rate)),
        ((state, BOLD),) if state else (),
        (),
    ]


def _shared_widths(columns: Columns) -> list[int]:
    """Return the cells each fixed field takes in a stack, gaps included."""
    sizes = (
        columns.percent,
        columns.count,
        columns.timing,
        columns.rate,
        columns.state,
    )
    return [*(size + _GAP if size else 0 for size in sizes), 0]


def _widths(fields: list[Sequence[Run]], shared: list[int]) -> list[int]:
    """Return the cells each field takes: its shared width, or more if it is wider."""
    return [
        max(slot, _width(runs) + _GAP if runs else 0)
        for slot, runs in zip(shared, fields, strict=True)
    ]


def _fit(widths: list[int], room: int) -> set[int]:
    """Choose the fields to keep within ``room`` cells, dropping by importance."""
    kept = {index for index, width in enumerate(widths) if width}
    used = sum(widths)
    for index in _DROP_ORDER:
        if used <= room:
            break
        kept.discard(index)
        used -= widths[index]
    return kept


def _keep_state(widths: list[int], kept: set[int], room: int) -> set[int]:
    """Keep the state label alone when it was dropped but fits in ``room``."""
    if not widths[_STATE] or _STATE in kept:
        return kept
    return {_STATE} if widths[_STATE] <= room else set()


_Placement = tuple[int, list[list[Sequence[Run]]]]


def _one_row(
    frame: Frame, fields: list[Sequence[Run]], columns: Columns, room: int
) -> _Placement:
    """Fit the bar, leaving room for states and a status reserve, beside its fields."""
    shared = _shared_widths(columns)
    widths = _widths(fields, shared)
    fits = _keep_state(shared, _fit(shared, room - _MIN_BAR), room)
    spare = room - sum(shared[i] for i in fits)
    free = spare - max(_STATE_SLOT - shared[_STATE], 0)
    wanted = _STATUS_RESERVE + _GAP if columns.status else 0
    reserve = min(wanted, free - _BAR_BEFORE_STATUS)
    if reserve < min(wanted, _MIN_STATUS + _GAP):
        reserve = 0
    bar_width = min(_MAX_BAR, max(_MIN_BAR, free - reserve), spare)
    kept = _keep_state(widths, _fit(widths, room - bar_width), room)
    fixed = sum(widths[i] for i in kept)
    bar_width = min(bar_width, room - fixed)
    dropped = any(widths[index] and index not in kept for index in range(_STATUS))
    pairs = iter(()) if dropped else _status_pairs(frame)
    if status := _fit_status(pairs, room - max(bar_width, 0) - fixed - _GAP):
        fields[_STATUS] = status
        kept.add(_STATUS)
    return bar_width, [[fields[index] for index in sorted(kept)]]


def _pack(
    widths: list[int], kept: list[int], area: int, height: int
) -> list[list[int]] | None:
    """Wrap fields ``kept`` in rows of ``area`` cells; None past ``height`` rows."""
    rows: list[list[int]] = []
    used = area
    for index in kept:
        if widths[index] > area:
            return None
        if used + widths[index] > area:
            if len(rows) == height:
                return None
            rows.append([])
            used = 0
        rows[-1].append(index)
        used += widths[index]
    return rows


def _status_grid(pairs: Iterator[_Pair], rows: int, width: int) -> list[list[Run]]:
    """Lay status pairs down ``rows`` rows, then across, in columns within ``width``."""
    lines: list[list[Run]] = [[] for _ in range(rows)]
    ends = [0] * rows
    room, left = width - _GAP, 0
    while column := list(islice(pairs, rows)):
        measured = [_pair(*pair) for pair in column]
        wide = max(width for width, _ in measured)
        if left + wide > room:
            if left or room < _MIN_STATUS:
                break
            measured = [(room, _cut_pair(*pair, room)) for pair in column]
            wide = room
        for row, (size, runs) in enumerate(measured):
            if left > ends[row]:
                lines[row].append((" " * (left - ends[row]), PLAIN))
            lines[row] += runs
            ends[row] = left + size
        left += wide + _GAP
    return lines


def _rows(
    frame: Frame, fields: list[Sequence[Run]], columns: Columns, room: int
) -> _Placement:
    """Fit the bar beside fields wrapped over its rows, with the status grid below."""
    height = frame.height
    widths = _widths(fields, _shared_widths(columns))
    fixed = [index for index in range(_STATUS) if widths[index]]
    total = sum(widths[index] for index in fixed)
    free = room - total - max(_STATE_SLOT - widths[_STATE], 0)
    bar_width = min(_MAX_BAR, max(_MIN_BAR, free), room)
    if fields[_STATE]:
        state = fields[_STATE][0][0].rstrip()
        fields[_STATE] = ((state, BOLD),) if state else ()
        widths[_STATE] = display_width(state) + _GAP if state else 0
        fixed = [index for index in fixed if widths[index]]
    drops = iter(_DROP_ORDER)
    while (packed := _pack(widths, fixed, room - bar_width, height)) is None:
        index = next(drops)
        fixed = [kept for kept in fixed if kept != index]
    if widths[_STATE] and _STATE not in fixed and widths[_STATE] <= room:
        bar_width = min(bar_width, room - widths[_STATE])
        packed = [[_STATE]]
    area = room - max(bar_width, 0)
    for kept in packed:
        if (runs := fields[kept[0]])[1:] and runs[0][0].isspace():
            fields[kept[0]] = (*runs[1:], runs[0])  # pad after the text, not before
    rows = [[fields[index] for index in kept] for kept in packed]
    rows += [[] for _ in range(height - len(rows))]
    if len(packed) < height:
        start, free = len(packed), area
    else:
        start, free = height - 1, area - sum(widths[index] for index in packed[-1])
    grid = _status_grid(_status_pairs(frame), height - start, free)
    for row, runs in enumerate(grid, start):
        if runs:
            rows[row].append(runs)
    return bar_width, rows


def layout(
    frame: Frame, width: int, style: Style, columns: Columns | None = None
) -> list[list[Run]]:
    """Lay out the rows of one bar, dropping fields that do not fit in ``width``.

    Args:
        frame: The bar snapshot to draw.
        width: Maximum display width in cells.
        style: Glyphs, palette, and color depth.
        columns: Shared field widths (default: this frame's own widths).

    Returns:
        Styled runs for each of the frame's ``height`` rows, from the top.

    Example:
        >>> from psnek._style import to_plain
        >>> from psnek._glyphs import ASCII
        >>> plain = Style(glyphs=ASCII, depth=ColorDepth.NONE)
        >>> frame = Frame(n=1, total=2, desc="x", elapsed=3, rate=0.5)
        >>> to_plain(layout(frame, 70, plain)[0])
        '| x #####_____   50.0%        1/2    0:03<0:02   2.00s/it'
    """
    label_cap = max(width // 3, _MIN_LABEL)
    if columns is None:
        columns = columns_for([frame], label_cap)
    texts = _texts(frame, label_cap)
    own = texts.label
    label = min(max(own, columns.label), max(label_cap, own))
    room = width - 2 - (label + 1 if label else 0)
    texts, columns = _fit_causes(frame, texts, columns, room)
    fields = _fields(frame, texts, columns)
    place = _one_row if frame.height == 1 else _rows
    bar_width, placed = place(frame, fields, columns, room)
    bar = _bar_cells(frame, bar_width, style) if bar_width > 0 else None
    indent = INDENT * len(frame.branches)
    lines = []
    for row, shown in enumerate(placed):
        if row:
            runs = _tree(frame, style, below=True)
        else:
            runs = [*_tree(frame, style), _spinner(frame, style), (" ", PLAIN)]
        if label:
            desc = texts.desc[row] if row < len(texts.desc) else ""
            gap = label - indent - display_width(desc) + 1
            runs.append((desc + " " * gap, PLAIN))
        if bar is not None:
            runs += bar[row]
        for field_runs in shown:
            runs.append(_SPACER)
            runs += field_runs
        while runs and not runs[-1][0].strip(_BLANKS):
            runs.pop()
        if runs:
            runs[-1] = (runs[-1][0].rstrip(_BLANKS), runs[-1][1])
        lines.append(clip(runs, width))
    return lines


def html_rows(rows: Sequence[str]) -> str:
    """Join a bar's HTML rows, set solid so Braille cells meet across rows.

    Example:
        >>> html_rows(["one"]), html_rows(["top", "bottom"]).endswith("bottom</span>")
        ('one', True)
    """
    return rows[0] if len(rows) == 1 else _SOLID + "<br>".join(rows) + "</span>"


def html_block(markup: str) -> str:
    """Wrap ``markup`` in a ``<pre>`` block styled as notebook bars are."""
    return f'<pre style="{PRE_CSS}">{markup}</pre>'


def _width(runs: Sequence[Run]) -> int:
    return sum(display_width(text) for text, _ in runs)


def summary(frame: Frame) -> str:
    """Render ``frame`` as one compact, uncolored line for logs, CI, and AI agents.

    Example:
        >>> summary(Frame(n=420, total=1000, elapsed=12, rate=33.6, desc="train"))
        'train: 42.0% 420/1000 0:12<0:17 33.6it/s'
    """
    desc, unit = _visible(frame.desc), _visible(frame.unit)
    parts = [f"{desc}:"] if desc else []
    if frame.fraction is not None:
        parts.append(_percent_text(frame).lstrip())
    parts += [_count_text(frame, unit).lstrip(), _timing_text(frame)]
    if frame.state != "running":
        parts.append(_state_text(frame))
    parts.append(format_rate(frame.rate, unit, frame.scale))
    parts += (f"{key}={value}" for key, value in _status_pairs(frame))
    return " ".join(parts)


def notice(frame: Frame) -> tuple[str, str]:
    """Return the title and body of a desktop notification about ``frame``.

    Example:
        >>> notice(Frame(n=3, total=4, elapsed=40, rate=0.1, desc="train",
        ...              state="failed"))
        ('train failed', '75.0% 3/4 0:40 failed 10.0s/it')
    """
    desc = truncate(_visible(frame.desc), _NOTICE_TITLE) or "progress"
    body = truncate(summary(replace(frame, desc="")), _NOTICE_BODY)
    return f"{desc} {frame.state}", body


def clip(runs: Sequence[Run], width: int) -> list[Run]:
    """Cut styled ``runs`` to at most ``width`` cells without an ellipsis."""
    joined = to_plain(runs)
    if EMOJI_PRESENTATION not in joined and display_width(joined) <= width:
        return list(runs)  # widths add up across runs without a selector
    clipped: list[Run] = []
    remaining = width
    for text, ink in runs:
        if (text_width := display_width(text)) > remaining:
            if remaining > 0:
                clipped.append((truncate(text, remaining, ellipsis=""), ink))
            break
        clipped.append((text, ink))
        remaining -= text_width
    return clipped


def panel_runs(line: PanelLine, width: int, style: Style) -> list[Run]:
    """Render a panel line after its gutter in ``width`` cells, cut with ``...``."""
    gutter = style.glyphs.gutter if line else style.glyphs.gutter.rstrip()
    room = max(width - display_width(gutter), 0)
    if isinstance(line, str):
        body = [(truncate(line, room), PLAIN)]
    elif _width(line) > room:
        body = [*clip(line, room - 3), ("...", PLAIN)]
    else:
        body = list(line)
    return clip([(gutter, FAINT), *body], width)


_LOG_LEVELS = tuple(range(logging.DEBUG, logging.CRITICAL + 1, 10))  # the stock levels
_ALARM = Ink(DEFAULT_PALETTE.failed, bold=True)
_LOG_STYLES: tuple[tuple[Run, Ink], ...] = (
    (("debug", FAINT), FAINT),
    (("info ", FAINT), PLAIN),
    (("warn ", Ink(DEFAULT_PALETTE.stalled, bold=True)), PLAIN),
    (("error", _ALARM), PLAIN),
    (("fatal", _ALARM), BOLD),
)


def log_lines(lines: Iterable[str], level: int, stamp: str = "") -> list[PanelLine]:
    """Tag raw ``lines`` with the time ``stamp`` and a level as salient as it is.

    Example:
        >>> first, second = log_lines(["saved", "shard 3"], logging.WARNING, "09:30:00")
        >>> to_plain(first), to_plain(second)
        ('09:30:00 warn  saved', '               shard 3')
    """
    tag, ink = _LOG_STYLES[max(bisect_right(_LOG_LEVELS, level) - 1, 0)]
    head = ((stamp + " ", FAINT), tag) if stamp else (tag,)
    first, *rest = (line.rstrip(_BLANKS) for line in clean_lines(lines))
    indent = " " * (_width(head) + 1)
    if not first:
        head = (*head[:-1], (tag[0].rstrip(), tag[1]))
    return [
        (*head, (f" {first}", ink)) if first else head,
        *(((indent, PLAIN), (line, ink)) if line else () for line in rest),
    ]


def log_stamp(when: float) -> str:
    """Format ``when``, in seconds since the epoch, as local ``HH:MM:SS``."""
    return time.strftime("%H:%M:%S", time.localtime(when))


def printed_runs(line: PanelLine, glyphs: Glyphs) -> list[Run]:
    """Return a panel ``line`` as printed above the bars, logged ones indented."""
    if isinstance(line, str):
        return [(line, PLAIN)]
    return [(" " * display_width(glyphs.gutter), PLAIN), *line] if line else []


def more_bars(
    hidden: int, width: int, kind: str = "more", noun: str = "bar"
) -> list[Run]:
    """Render the faint line counting ``hidden`` bars (or other ``noun``) not shown."""
    plural = noun if hidden == 1 else f"{noun}s"
    return clip([(f"  +{hidden} {kind} {plural}", FAINT)], width)


_GIB, _MIB = 2**30, 2**20
_LOSS = (("loss", "0.137"), ("lr", "3e-4"))
_LONG = (("note", "a very long status value " * 3),)
GALLERY: tuple[tuple[str, Frame], ...] = (
    ("empty", Frame(0, 100, desc="empty")),
    ("start", Frame(3, 1000, 0.4, 7.5, "start")),
    ("half", Frame(50, 100, 12.0, 4.2, "half", phase=0.3)),
    ("bytes", Frame(1.5 * _GIB, 4 * _GIB, 95.0, 16.4 * _MIB, "download", "B", "iec")),
    ("status", Frame(420, 1000, 12.5, 33.6, "train", status=_LOSS, phase=0.7)),
    ("unknown", Frame(1234, None, 8.0, 154.2, "stream", phase=0.8)),
    ("si rate", Frame(3.2e6, 1e7, 2.7, 1.18e6, "tokens", "tok", "si", phase=0.5)),
    ("rough eta", Frame(37, 100, 60, 0.9, "bursty", approximate=True, phase=0.4)),
    (
        "stalled",
        Frame(37, 100, 60, 0.9, "slow", state="stalled", idle=9, approximate=True),
    ),
    (
        "stuck",
        Frame(37, 100, 99, 0.6, "stuck", state="stuck", idle=48, approximate=True),
    ),
    ("done", Frame(100, 100, 29.3, 3.4, "done", state="done")),
    ("failed", Frame(812, 1200, 41, 19.8, "failed", state="failed")),
    ("interrupted", Frame(3, 10, 7, 0.4, "cancel", state="interrupted")),
    ("slow rate", Frame(2, 9, 9.0, 0.25, "slow items")),
    ("minutes", Frame(3, 20, 1210, 1 / 400, "renders")),
    ("hours", Frame(1, 6, 9500, 1 / 9000, "sims")),
    ("unknown done", Frame(880, None, 6.0, 146.7, "stream", state="done")),
    (
        "unknown stall",
        Frame(512, None, 75, 6.8, "stream", state="stalled", idle=14, phase=0.8),
    ),
    (
        "unknown stuck",
        Frame(40, None, 300, 0.2, "stream", state="stuck", idle=61, phase=1),
    ),
    ("unknown fail", Frame(96, None, 20, 4.8, "stream", state="failed", phase=1.9)),
    ("unknown cancel", Frame(12, None, 3, 4.0, "stream", state="interrupted")),
    ("overshoot", Frame(130, 100, 4.0, 32.5, "over")),
    ("zero total", Frame(0, 0, desc="nothing", state="done")),
    ("huge", Frame(7, 10**12, 3.0, 2.3, "huge")),
    ("days", Frame(6100, 10000, 264_225, 0.0231, "days")),
    ("wide desc", Frame(3, 8, 2.0, 1.5, "日本語の説明")),
    ("long status", Frame(42, 100, 8.0, 5.2, "train", status=_LONG)),
    ("nested", Frame(12, 40, 3.0, 4.1, "batch", phase=0.4, branches=(True,))),
    ("nested last", Frame(5, 8, 1.0, 5.2, "shard", phase=1.1, branches=(True, False))),
)


def _coil(style: Style, height: int) -> list[list[Run]]:
    """Draw a column ``height`` rows tall at every fill step, from empty to full."""
    gradient, steps = style.palette.gradient, STEPS * height
    floors = [(_floor(style, row, height), FAINT) for row in range(height)]
    rows: list[list[Run]] = [[floor, (" ", PLAIN)] for floor in floors]
    for step in range(1, steps + 1):
        ink = Ink(gradient_at(gradient, step / steps))
        for row, level in enumerate(_levels(height, step)):
            run = (style.glyphs.partials[level], ink) if level else floors[row]
            rows[row] += [run, (" ", PLAIN)]
    return rows


def gallery(
    width: int = 100,
    style: Style | None = None,
    html: bool = False,
    phase: float = 0.0,
    height: int = 1,
) -> str:
    """Render every reference state, for design review and snapshot tests.

    Args:
        width: Line width in cells.
        style: Glyphs, palette, and color depth (defaults to Braille truecolor).
        html: Return an HTML ``<pre>`` block instead of ANSI text.
        phase: Animation clock offset applied to every frame.
        height: Rows each bar spans.

    Returns:
        A legend of the fill steps, then each state's rows, labeled.

    Raises:
        ValueError: If ``height`` is below one.
    """
    style = Style() if style is None else style
    name_width = max(len(name) for name, _ in GALLERY) + 2
    frames = [
        replace(frame, phase=frame.phase + phase, height=height) for _, frame in GALLERY
    ]
    room = width - name_width
    cap = max(room // 3, _MIN_LABEL)
    label = columns_for(frames, cap).label
    shown = [("coil", _coil(style, height))]
    for (name, _), frame in zip(GALLERY, frames, strict=True):
        columns = replace(columns_for([frame], cap), label=label)
        shown.append((name, layout(frame, room, style, columns)))
    lines = []
    for name, rows in shown:
        titles = [name.rjust(name_width - 2) + "  "] + [" " * name_width] * len(rows)
        titled = [
            clip([(title, FAINT), *row], width)
            for title, row in zip(titles, rows, strict=False)
        ]
        if html:
            lines.append(html_rows([to_html(runs, style.depth) for runs in titled]))
        else:
            lines += (to_ansi(runs, style.depth) for runs in titled)
    return html_block("\n".join(lines)) if html else "\n".join(lines)
