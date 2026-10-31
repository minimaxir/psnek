"""Text utilities: sanitization, terminal line replay, width, and formatting."""

import math
import re
import unicodedata
from collections.abc import Iterable, Iterator
from functools import lru_cache
from typing import Literal, get_args

Scale = Literal["none", "si", "iec"]
SCALES: tuple[Scale, ...] = get_args(Scale)

ESCAPE_SEQUENCE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[\]PX^_][^\x07\x1b]*(?:\x07|\x1b\\)|[ -/]*[0-~])"
)
_LINE_CONTROL = re.compile(r"\r|\x1b\[([012]?)K|\x1b\[([0-9]*)G")
_CURSOR_CONTROL = re.compile(f"{_LINE_CONTROL.pattern}|{ESCAPE_SEQUENCE.pattern}|\t")
_NARROW = re.compile("[\x00-\x7f\u2500-\u257f\u2800-\u28ff]*")  # one cell each
_DISPUTED = re.compile(
    "[^\x20-\x7e\u2800-\u28ff\u3001\u3002\u300c-\u300f\u3041-\u3096\u30a1-\u30fe"
    "\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7a3\uff01-\uff5e]"
)
_SI_PREFIXES = ("", "k", "M", "G", "T", "P", "E", "Z", "Y")
_IEC_PREFIXES = ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi", "Yi")
_ZERO_WIDTH_CATEGORIES, _WIDE, _WIDE_CELLS, _PAIR = ("Mn", "Me"), ("W", "F"), 2, 2

MAX_MARKS, TAB_STOP, _MAX_COLUMN, _BMP_END = 8, 8, 1 << 12, 0xFFFF
EMOJI_PRESENTATION = "\ufe0f"
_JOINERS = frozenset("\u200c\u200d")
_KEPT_CONTROLS = _JOINERS | {"\u00ad", "\t"}
_ROUNDING_LIMITS, _HALF = (9.995, 99.95), 0.5
_PICTOGRAPHS = range(0x1F300, 0x1FB00)
_FIXED_RATE, _SCALED_RATE = 100, 10_000
_MINUTE, _HOUR, _DAY = 60, 3600, 86400
_MAX_DURATION, _MAX_AGE = 100 * _DAY, 100 * _HOUR
_TIME_STEPS = ((1, "s", _MINUTE), (_MINUTE, "min", _MINUTE), (_HOUR, "h", 1000))


def is_text_pair(item: object) -> bool:
    """Return whether ``item`` is a tuple of two strings."""
    if not isinstance(item, tuple) or len(item) != _PAIR:
        return False
    return isinstance(item[0], str) and isinstance(item[1], str)


def sanitize(text: str) -> str:
    r"""Make one line of untrusted ``text`` safe inside a terminal frame or HTML.

    Escape sequences are removed (an unterminated one loses its introducer),
    tabs and other spaces become spaces, and other non-printable code points
    except joiners and the soft hyphen become visible ``\\uXXXX`` escapes.

    Example:
        >>> sanitize("\x1b[31mred\x1b[0m\tok\u202e")
        'red     ok\\u202e'
    """
    if text.isprintable():
        return text
    if "\t" in text and text.replace("\t", "").isprintable():
        return _expand_tabs(text)
    text = ESCAPE_SEQUENCE.sub("", text)
    if not text.isprintable():
        hidden = {char for char in {*text} if not char.isprintable()} - _KEPT_CONTROLS
        text = text.translate({ord(char): _visible(char) for char in hidden})
    return _expand_tabs(text) if "\t" in text else text


def _expand_tabs(text: str) -> str:
    if text.isascii():
        return text.expandtabs(TAB_STOP)
    *parts, last = text.split("\t")
    pieces: list[str] = []
    column = 0
    for part in parts:
        column += display_width(part)
        gap = TAB_STOP - column % TAB_STOP
        pieces += (part, " " * gap)
        column += gap
    pieces.append(last)
    return "".join(pieces)


def _visible(char: str) -> str:
    if unicodedata.category(char) == "Zs":
        return " " * char_width(char)
    code = ord(char)
    return f"\\u{code:04x}" if code <= _BMP_END else f"\\U{code:08x}"


def _put(cells: list[str], text: str, column: int) -> int:
    """Write sanitized ``text`` over ``cells`` from ``column``; return the end."""
    if not text:
        return column
    cells.extend(" " * (column - len(cells)))
    if text.isascii():
        end = column + len(text)
        if column < len(cells) and cells[column] == "":
            cells[column - 1] = " "
        if end < len(cells) and cells[end] == "":
            cells[end] = " "
        cells[column:end] = text
        return end
    for char in text:
        width = char_width(char)
        if char == EMOJI_PRESENTATION and column and cells[column - 1] == "":
            width = 0
        if width == 0:
            if column:
                head = column - 1 - (cells[column - 1] == "")
                if len(cells[head]) <= MAX_MARKS:
                    cells[head] += char
            continue
        if column < len(cells) and cells[column] == "":
            cells[column - 1] = " "
        stop = column + width
        if stop < len(cells) and cells[stop] == "":
            cells[stop] = " "
        cells[column:stop] = [char, *[""] * (width - 1)]
        column = stop
    return column


def _erase(cells: list[str], mode: str, column: int) -> None:
    """Apply erase-in-line ``mode`` (``""``/``0``, ``1``, or ``2``) at ``column``."""
    if mode == "2":
        cells.clear()
    elif mode == "1":
        stop = min(column + 1, len(cells), _MAX_COLUMN)
        cells[:stop] = " " * stop
        if stop < len(cells) and cells[stop] == "":
            cells[stop] = " "
    else:
        if 0 < column < len(cells) and cells[column] == "":
            cells[column - 1] = " "
        del cells[column:]


def _overwrite(line: str) -> str:
    cells: list[str] = []
    column = position = 0
    for match in _CURSOR_CONTROL.finditer(line):
        column = _put(cells, sanitize(line[position : match.start()]), column)
        position = match.end()
        erase, move = match.groups()
        if erase is not None:
            _erase(cells, erase, column)
        elif move is not None:  # CSI digits G, clamped without parsing huge numbers
            digits = move.lstrip("0")[: len(str(_MAX_COLUMN)) + 1]
            column = min(int(digits or 1), _MAX_COLUMN) - 1
        elif match.group() == "\t":
            stop = min((column // TAB_STOP + 1) * TAB_STOP, _MAX_COLUMN - 1)
            column = max(column, stop)
        elif match.group() == "\r":
            column = 0
    _put(cells, sanitize(line[position:]), column)
    return "".join(cells)


def clean_lines(lines: Iterable[str]) -> list[str]:
    r"""Replay carriage returns in raw ``lines`` as a terminal would, then sanitize.

    Example:
        >>> clean_lines(["hello\rab", "10%\r\x1b[2K20%", "crlf\r", "日本\rxy"])
        ['abllo', '20%', 'crlf', 'xy本']
    """
    return [
        _overwrite(raw) if _LINE_CONTROL.search(raw) else sanitize(raw)
        for raw in (line.removesuffix("\r") for line in lines)
    ]


def join_lines(lines: Iterable[str]) -> str:
    """Return ``lines`` joined into text, each ending with a newline."""
    return "".join(f"{line}\n" for line in lines)


@lru_cache(maxsize=4096)
def char_width(char: str) -> int:
    """Return the cells code point ``char`` occupies: 0, 1, or 2 (wide, emoji)."""
    if char == EMOJI_PRESENTATION:
        return 1
    if char in _JOINERS or unicodedata.category(char) in _ZERO_WIDTH_CATEGORIES:
        return 0
    if unicodedata.east_asian_width(char) in _WIDE or ord(char) in _PICTOGRAPHS:
        return _WIDE_CELLS
    return 1


def display_width(text: str) -> int:
    """Return the terminal cells that sanitized ``text`` occupies."""
    if text.isascii():
        return len(text)
    if len(text) == 1:
        return char_width(text)
    if _NARROW.fullmatch(text):
        return len(text)
    if EMOJI_PRESENTATION in text:
        return sum(_cells(text))
    return sum(map(char_width, text))


def disputed(text: str) -> bool:
    """Whether terminals may draw sanitized ``text`` at another width than psnek.

    Example:
        >>> disputed("\u251c\u2500 load \u28ff"), disputed("ok")
        (True, False)
    """
    return not text.isascii() and _DISPUTED.search(text) is not None


def width_bounds(text: str) -> tuple[int, int]:
    """Return the fewest and most cells terminals may draw sanitized ``text`` in.

    Example:
        >>> width_bounds("caf\u00e9"), width_bounds("\u65e5\u672c")
        ((3, 5), (4, 4))
    """
    width = display_width(text)
    if not disputed(text):
        return width, width
    least = most = width
    for char, cells in zip(text, _cells(text), strict=True):
        if _DISPUTED.match(char):
            least -= cells
            most += _WIDE_CELLS - cells
    return least, most


def ends_at_margin(text: str, columns: int) -> bool:
    """Whether sanitized ``text``, printed with autowrap, fills its last row.

    Example:
        >>> ends_at_margin("x" * 8, 4), ends_at_margin("xxx日", 4)
        (True, False)
    """
    width = display_width(text)
    if width < columns:
        return False
    if text.isascii() or _NARROW.fullmatch(text):
        return width % columns == 0
    column = 0
    for cells in _cells(text):
        if column + cells > columns:
            column = 0
        column += cells
    return column == columns


def _cells(text: str) -> Iterator[int]:
    """Yield the cells each code point adds; a selector after a wide one adds 0."""
    wide = False
    for char in text:
        cells = char_width(char)
        if char == EMOJI_PRESENTATION and wide:
            cells = 0
        elif cells:
            wide = cells == _WIDE_CELLS
        yield cells


def pad(text: str, width: int, left: bool = True) -> str:
    """Pad sanitized ``text`` with spaces to ``width`` cells, on the ``left`` side.

    Example:
        >>> pad("画像", 6)
        '  画像'
    """
    fill = " " * max(width - display_width(text), 0)
    return fill + text if left else text + fill


def truncate(text: str, width: int, ellipsis: str = "...") -> str:
    """Truncate sanitized ``text`` to ``width`` cells, ending in ASCII ``ellipsis``."""
    budget = width - len(ellipsis)
    if text.isascii():
        if len(text) <= width:
            return text
        return text[: max(budget, 0)] + ellipsis[: max(width, 0)]
    used, cut, marks = 0, 0, 0
    for index, cells in enumerate(_cells(text)):
        used += cells
        marks = 0 if cells else marks + 1
        if used > width or marks > MAX_MARKS:
            return text[:cut] + ellipsis[: max(width, 0)]
        if used <= budget:
            cut = index + 1
    return text


def wrap(text: str, width: int, rows: int) -> tuple[str, ...]:
    """Wrap sanitized ``text`` at spaces into at most ``rows`` lines of ``width``.

    Example:
        >>> wrap("resize every image in the folder", 12, 2)
        ('resize every', 'image in ...')
    """
    lines: list[str] = []
    for word in text.split():
        joined = f"{lines[-1]} {word}" if lines else word
        if lines and display_width(joined) <= width:
            lines[-1] = joined
            continue
        rest = word
        while rest and len(lines) <= rows:
            piece = truncate(rest, width, ellipsis="")
            lines.append(piece)
            rest = rest[len(piece) :]
        if len(lines) > rows:
            break
    if len(lines) <= rows:
        return tuple(lines)
    return (*lines[: rows - 1], truncate(f"{lines[rows - 1]} {lines[rows]}", width))


def _three_digits(value: float) -> str:
    digits = 2 - sum(abs(value) >= limit for limit in _ROUNDING_LIMITS)
    return f"{value:.{digits}f}"


def format_number(value: float, scale: Scale = "none") -> str:
    """Format ``value`` with an SI (1000) or IEC (1024) prefix per ``scale``.

    Example:
        >>> format_number(1234567, "si"), format_number(1536, "iec")
        ('1.23M', '1.50Ki')
    """
    if scale == "none":
        return f"{value:.0f}" if float(value).is_integer() else f"{value:.2f}"
    base, prefixes = (
        (1000.0, _SI_PREFIXES) if scale == "si" else (1024.0, _IEC_PREFIXES)
    )
    magnitude = 0
    while abs(value) >= base - _HALF and magnitude < len(prefixes) - 1:
        value /= base
        magnitude += 1
    if abs(value) >= base - _HALF:
        return f"{'<-' if value < 0 else '>'}999{prefixes[-1]}"
    if magnitude == 0 and float(value).is_integer():
        return f"{value:.0f}"
    return f"{_three_digits(value)}{prefixes[magnitude]}"


def format_value(value: object) -> str:
    """Format floats to four significant digits, keeping trailing zeros; else ``str``.

    Example:
        >>> format_value(0.02439), format_value(0.02), format_value(42)
        ('0.02439', '0.02000', '42')
    """
    return f"{value:#.4g}" if isinstance(value, float) else str(value)


def format_duration(seconds: float | None) -> str:
    """Format ``seconds`` as ``m:ss``, ``h:mm:ss``, ``NdHHh``, or ``--:--`` if unknown.

    Example:
        >>> format_duration(75), format_duration(3725), format_duration(266400)
        ('1:15', '1:02:05', '3d02h')
    """
    if seconds is None or not 0 <= seconds < math.inf:
        return "--:--"
    if seconds >= _MAX_DURATION:
        return "99d+"
    whole = int(seconds)
    if whole >= _DAY:
        days, rest = divmod(whole, _DAY)
        return f"{days}d{rest // _HOUR:02d}h"
    minutes, secs = divmod(whole, _MINUTE)
    hours, minutes = divmod(minutes, _MINUTE)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def format_rate(rate: float | None, unit: str, scale: Scale = "none") -> str:
    """Format ``rate`` per second in ``unit``, as time per unit below one.

    Example:
        >>> format_rate(12.34, "it"), format_rate(54321, "it"), format_rate(0.004, "it")
        ('12.3it/s', '54.3kit/s', '4.17min/it')
    """
    if rate is None or not 0 < rate < math.inf:
        return f"--{unit}/s"
    if rate < 1:  # pick the unit after rounding, so 59.96 seconds reads 1.00min
        for size, suffix, limit in _TIME_STEPS:
            text = _three_digits(1 / rate / size)
            if float(text) < limit:
                return f"{text}{suffix}/{unit}"
        return f">999h/{unit}"
    if scale == "none" and round(rate) < _SCALED_RATE:
        if round(rate, 1) < _FIXED_RATE:
            return f"{rate:.1f}{unit}/s"
        return f"{rate:.0f}{unit}/s"
    return f"{format_number(rate, 'si' if scale == 'none' else scale)}{unit}/s"


def format_age(seconds: float) -> str:
    """Format an idle time of ``seconds`` compactly, always with a unit.

    Example:
        >>> format_age(9.4), format_age(312), format_age(7380)
        ('9s', '5m12s', '2h03m')
    """
    if not math.isfinite(seconds):
        return "--"
    if seconds >= _MAX_AGE:
        return "99h+"
    whole = int(max(seconds, 0))
    if whole < _MINUTE:
        return f"{whole}s"
    minutes, secs = divmod(whole, _MINUTE)
    if minutes < _MINUTE:
        return f"{minutes}m{secs:02d}s"
    return f"{minutes // _MINUTE}h{minutes % _MINUTE:02d}m"


def encodable(text: str, encoding: str) -> str:
    r"""Replace characters of ``text`` that ``encoding`` lacks with backslash escapes.

    Example:
        >>> encodable("caf\u00e9 \u28ff", "ascii")
        'caf\\xe9 \\u28ff'
    """
    return text.encode(encoding, "backslashreplace").decode(encoding)
