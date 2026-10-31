"""Panel content laid out as sanitized text lines or borderless, aligned tables."""

import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from itertools import groupby, islice, zip_longest
from operator import itemgetter
from typing import Literal

from psnek._style import BOLD, FAINT, PLAIN, PanelLine, Run
from psnek._text import clean_lines, display_width, format_value, sanitize, truncate

Justify = Literal["left", "right", "center"]
CELL_WIDTH, _GAP = 40, "  "

_TEXT_TYPES = (str, bytes, bytearray)
_MISSING: Run = ("", PLAIN)


def as_record(item: object) -> Mapping[object, object] | None:
    """Return ``item`` as a mapping: itself, a dataclass's fields, or a named tuple."""
    if isinstance(item, Mapping):
        return item
    if is_dataclass(item) and not isinstance(item, type):
        return {
            field.name: getattr(item, field.name)
            for field in fields(item)
            if field.repr
        }
    if isinstance(item, tuple) and hasattr(item, "_asdict"):
        return dict(item._asdict())
    return None


def _frame_records(
    content: object, limit: int
) -> tuple[list[Mapping[object, object]], int] | None:
    """Return a loaded polars or pandas DataFrame's first ``limit`` rows and height."""
    polars = sys.modules.get("polars")
    if polars is not None and isinstance(content, polars.DataFrame):
        return content.head(limit).to_dicts(), content.height
    pandas = sys.modules.get("pandas")
    if pandas is not None and isinstance(content, pandas.DataFrame):
        head = content.head(limit)
        if not isinstance(head.index, pandas.RangeIndex):
            head = head.reset_index(allow_duplicates=True)
        return head.to_dict("records"), len(content)
    return None


def _cell(value: object) -> Run:
    return ("None", FAINT) if value is None else (sanitize(format_value(value)), PLAIN)


def _shown(count: int, limit: int, reserved: int = 0) -> int:
    """Return how many of ``count`` rows fit, keeping a line for the overflow note."""
    room = limit - reserved
    return count if count <= room else max(room - 1, 0)


def _more(hidden: int, noun: str) -> list[PanelLine]:
    return [((f"... {hidden} more {noun}", FAINT),)] if hidden > 0 else []


def _align(text: str, width: int, justify: Justify, last: bool) -> str:
    fill = width - display_width(text)
    before = {"left": 0, "right": fill, "center": fill // 2}[justify]
    return " " * before + text + ("" if last else " " * (fill - before) + _GAP)


def _line(row: list[Run], widths: list[int], justify: Justify) -> PanelLine:
    """Join a row's aligned cells into runs, without trailing blanks or padding."""
    end = len(row)
    while end and not row[end - 1][0]:
        end -= 1
    cells = [
        (_align(text, widths[index], justify, index == end - 1), ink)
        for index, (text, ink) in enumerate(row[:end])
    ]
    return tuple(
        ("".join(text for text, _ in group), ink)
        for ink, group in groupby(cells, key=itemgetter(1))
    )


def _table(rows: list[list[Run]], hidden: int, justify: Justify) -> list[PanelLine]:
    """Align ``rows`` of cells into columns; all but the last are cut to fit."""
    columns = max(map(len, rows), default=0)
    rows = [
        [
            (truncate(text, CELL_WIDTH) if index < columns - 1 else text, ink)
            for index, (text, ink) in enumerate(row)
        ]
        for row in rows
    ]
    texts = ([text for text, _ in row] for row in rows)
    widths = [
        max(map(display_width, cells)) for cells in zip_longest(*texts, fillvalue="")
    ]
    return [_line(row, widths, justify) for row in rows] + _more(hidden, "rows")


def _records_table(
    records: list[Mapping[object, object]], count: int, limit: int, justify: Justify
) -> list[PanelLine]:
    """Lay out the first of ``count`` records under a bold header of their keys."""
    records = records[: _shown(count, limit, reserved=1)]
    if not records:
        return _more(count, "rows")
    keys = list(dict.fromkeys(key for record in records for key in record))
    header = [(sanitize(str(key)), BOLD) for key in keys]
    body = [
        [_cell(record[key]) if key in record else _MISSING for key in keys]
        for record in records
    ]
    return _table([header, *body], count - len(records), justify)


def _sequence_table(
    items: Sequence[object], limit: int, justify: Justify
) -> list[PanelLine]:
    count = len(items)
    shown = list(islice(items, _shown(count, limit)))
    records = [record for item in shown if (record := as_record(item)) is not None]
    if shown and len(records) == len(shown):
        return _records_table(records, count, limit, justify)
    rows = [
        [_cell(part) for part in item]
        if isinstance(item, Sequence) and not isinstance(item, _TEXT_TYPES)
        else [_cell(item)]
        for item in shown
    ]
    return _table(rows, count - len(shown), justify)


def panel_lines(
    content: object, limit: int, justify: Justify = "left"
) -> list[PanelLine]:
    """Lay out panel ``content`` (see `Bar.panel`) in at most ``limit`` lines.

    Example:
        >>> lines = panel_lines({"loss": 0.1234567, "step": 30}, limit=8)
        >>> ["".join(text for text, _ in line) for line in lines]
        ['loss  0.1235', 'step  30']
    """
    if content is None or limit <= 0:
        return []
    record = as_record(content)
    if record is not None:
        shown = _shown(len(record), limit)
        rows = [
            [(sanitize(str(key)), BOLD), _cell(value)]
            for key, value in islice(record.items(), shown)
        ]
        return _table(rows, len(record) - shown, justify)
    frame = _frame_records(content, limit)
    if frame is not None:
        return _records_table(*frame, limit, justify)
    if isinstance(content, Sequence) and not isinstance(content, _TEXT_TYPES):
        return _sequence_table(content, limit, justify)
    text = str(content)
    parts = text.split("\n") if text else []
    shown = _shown(len(parts), limit)
    return [*clean_lines(parts[:shown]), *_more(len(parts) - shown, "lines")]
