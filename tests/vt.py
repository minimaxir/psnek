"""A minimal, strict VT screen model for asserting what a live region leaves.

It understands exactly the sequences psnek emits (text, LF, CPL, CHA, CUF, EL, ED,
SGR with the attributes psnek uses, the DEC private modes psnek toggles, OSC
9;4 progress, and the OSC 9, 777, and 99 notifications and BEL, which leave
the screen as it was) and raises on anything else, so an unexpected sequence, a
control or format character (other than joiners and the soft hyphen), a
cursor move into scrollback, or a line overflowing with autowrap off fails
the test.
"""

import re
import unicodedata
from collections.abc import Callable

from psnek._text import char_width

_TOKEN = re.compile(
    r"\x1b\[(?P<params>[0-9;?]*)(?P<final>[@-~])"
    r"|(?P<osc>\x1b\]9;4;[0-4];[0-9]{1,3}\x1b\\)"
    r"|(?P<notice>\x1b\](?:9;(?!4;)|777;notify;[^\x1b;]*;|99;[a-z0-9=:]+;)[^\x1b]*\x1b\\)"
    r"|(?P<bell>\x07)"
    r"|(?P<text>[^\x1b\x07]+)"
)
_PRIVATE_MODES = {"?25h", "?25l", "?7h", "?7l", "?2026h", "?2026l"}
_SGR = re.compile(
    r"(?:0|1|2|38;2;(?:\d{1,3};){2}\d{1,3}|38;5;\d{1,3})"
    r"(?:;(?:0|1|2|38;2;(?:\d{1,3};){2}\d{1,3}|38;5;\d{1,3}))*"
)
_REJECTED_CATEGORIES = ("Cc", "Cf")
_ALLOWED_FORMAT = frozenset("\u200c\u200d\u00ad")


Pen = tuple[bool, bool, str]
_PLAIN_PEN: Pen = (False, False, "")


class Screen:
    """Rows of cells plus a cursor; rows above ``rows`` become scrollback.

    A wide character occupies its cell and an empty continuation cell, and a
    zero-width character joins the cell before it. Each cell also records the
    pen it was written with: bold, faint, and the SGR color parameters.
    """

    def __init__(
        self,
        columns: int = 80,
        rows: int = 24,
        width: Callable[[str], int] = char_width,
    ) -> None:
        """Create an empty screen with autowrap on.

        Args:
            columns: Screen width in cells.
            rows: Visible screen height in rows.
            width: The cells this terminal draws each code point in, which
                may differ from psnek's estimate.
        """
        self.columns = columns
        self.width = width
        self.rows = rows
        self.lines: list[list[str]] = [[]]
        self.pens: list[list[Pen]] = [[]]
        self.pen = _PLAIN_PEN
        self.row = 0
        self.column = 0
        self.wrap = True

    @property
    def top(self) -> int:
        """Index of the first visible row."""
        return max(len(self.lines) - self.rows, 0)

    def feed(self, data: str) -> None:
        """Apply terminal output.

        Args:
            data: Text and escape sequences.

        Raises:
            ValueError: On anything psnek should never emit.
        """
        position = 0
        while position < len(data):
            match = _TOKEN.match(data, position)
            if match is None:
                raise ValueError(f"unexpected escape at {data[position:][:12]!r}")
            if match["text"] is not None:
                self._text(match["text"])
            elif match["final"] is not None:
                self._csi(match["params"], match["final"])
            position = match.end()

    def _newline(self) -> None:
        self.row += 1
        self.column = 0
        if self.row == len(self.lines):
            self.lines.append([])
            self.pens.append([])

    def _text(self, text: str) -> None:
        for char in text:
            if char == "\n":
                self._newline()
                continue
            rejected = unicodedata.category(char) in _REJECTED_CATEGORIES
            if rejected and char not in _ALLOWED_FORMAT:
                raise ValueError(f"control character {char!r} in output")
            width = self.width(char)
            line = self.lines[self.row]
            if width == 0:
                if not 0 < self.column <= len(line):
                    raise ValueError(f"zero-width {char!r} without a base cell")
                head = self.column - 1
                if head and not line[head]:
                    head -= 1
                line[head] += char
                continue
            if self.column + width > self.columns:
                if not self.wrap:
                    raise ValueError("line overflowed with autowrap off")
                self._newline()
                line = self.lines[self.row]
            end = self.column + width
            if 0 < self.column < len(line) and not line[self.column]:
                line[self.column - 1] = " "
            if end < len(line) and not line[end]:
                line[end] = " "
            pens = self.pens[self.row]
            pens.extend([_PLAIN_PEN] * (end - len(line)))
            line.extend(" " * (end - len(line)))
            line[self.column : end] = [char, *[""] * (width - 1)]
            pens[self.column : end] = [self.pen] * width
            self.column = end

    def _csi(self, params: str, final: str) -> None:
        if final == "m":
            if params and not _SGR.fullmatch(params):
                raise ValueError(f"unexpected SGR parameters {params!r}")
            if any(int(value) > 255 for value in params.split(";") if value):
                raise ValueError(f"SGR parameter out of range in {params!r}")
            self._sgr(params.split(";"))
            return
        if params.startswith("?"):
            if params + final not in _PRIVATE_MODES:
                raise ValueError(f"unexpected private mode {params}{final}")
            if params == "?7":
                self.wrap = final == "h"
            return
        count = int(params or 1) or 1
        line = self.lines[self.row]
        if final == "F":
            if self.row - count < self.top:
                raise ValueError("cursor moved above the visible screen")
            self.row -= count
            self.column = 0
        elif final == "G":
            self.column = min(count, self.columns) - 1
        elif final == "C":
            self.column = min(self.column + count, self.columns - 1)
        elif final in "KJ" and not params:
            self._erase(line, below=final == "J")
        else:
            raise ValueError(f"unexpected CSI {params}{final}")

    def _sgr(self, codes: list[str]) -> None:
        """Update the pen with SGR ``codes``, already checked against `_SGR`."""
        bold, faint, color = self.pen
        while codes:
            code = codes.pop(0)
            if code in ("", "0"):
                bold, faint, color = _PLAIN_PEN
            elif code == "1":
                bold = True
            elif code == "2":
                faint = True
            else:
                size = 2 if codes[0] == "5" else 4
                color = ";".join(codes[:size])
                del codes[:size]
        self.pen = (bold, faint, color)

    def _erase(self, line: list[str], below: bool) -> None:
        """Blank from the cursor to the end of the line, and the rows below.

        At the right margin, waiting to wrap, the cursor is on the last cell,
        so that cell is blanked too, as real terminals do.
        """
        column = min(self.column, self.columns - 1)
        if 0 < column < len(line) and not line[column]:
            line[column - 1] = " "
        del line[column:]
        del self.pens[self.row][column:]
        if below:
            for row, pens in zip(
                self.lines[self.row + 1 :], self.pens[self.row + 1 :], strict=True
            ):
                row.clear()
                pens.clear()

    def styled(
        self, start: int = 0, stop: int | None = None
    ) -> list[list[tuple[str, Pen]]]:
        """Return rows ``start`` to ``stop`` as cells with their pens.

        A blank cell's pen is ignored, as a space shows no color or weight,
        and trailing blank cells are removed.

        Returns:
            Each row's cells as text and pen.
        """
        rows = []
        for line, pens in zip(
            self.lines[start:stop], self.pens[start:stop], strict=True
        ):
            cells = [
                (char, pen if char.strip() else _PLAIN_PEN)
                for char, pen in zip(line, pens, strict=True)
            ]
            while cells and not cells[-1][0].strip():
                cells.pop()
            rows.append(cells)
        return rows

    def text(self) -> list[str]:
        """Return every row (scrollback included) with trailing spaces removed.

        Returns:
            The rows, without trailing empty rows.
        """
        rows = ["".join(line).rstrip() for line in self.lines]
        while rows and not rows[-1]:
            rows.pop()
        return rows
