"""psnek: a fast, modern progress bar with a Braille snake.

Example:
    >>> import psnek
    >>> for item in psnek.track(range(3), "work", options=psnek.Options(disable=True)):
    ...     pass
"""

import importlib.metadata

from psnek._async import amap, gather
from psnek._bar import Bar, LogHandler, Options, PanelPosition, pmap, write
from psnek._frame import Frame, Health, Outcome, State
from psnek._glyphs import ASCII, BRAILLE, Glyphs, Style
from psnek._render import Columns, columns_for, gallery, html_block, html_rows, layout
from psnek._style import ColorDepth, Palette, Run, to_ansi, to_html
from psnek._table import Justify
from psnek._text import Scale
from psnek._track import AsyncTrack, Track, TrackArgs, atrack, track, zip

# fmt: off
__all__ = [
    "ASCII", "BRAILLE", "AsyncTrack", "Bar", "ColorDepth", "Columns", "Frame",
    "Glyphs", "Health", "Justify", "LogHandler", "Options", "Outcome", "Palette",
    "PanelPosition", "Run", "Scale", "State", "Style", "Track", "TrackArgs", "amap",
    "atrack", "columns_for", "gallery", "gather", "layout", "pmap", "render",
    "to_ansi", "track", "write", "zip",
]
# fmt: on


def __getattr__(name: str) -> str:
    """Resolve ``psnek.__version__`` lazily from the installed metadata.

    Raises:
        AttributeError: For any other ``name``, or when psnek is not installed.
    """
    if name == "__version__":
        try:
            return importlib.metadata.version("psnek")
        except importlib.metadata.PackageNotFoundError as error:
            raise AttributeError("psnek is not installed; no version") from error
    raise AttributeError(f"module 'psnek' has no attribute {name!r}")


def render(
    frame: Frame, width: int = 80, style: Style | None = None, html: bool = False
) -> str:
    """Render any bar state to ANSI text (or HTML), ignoring environment variables.

    Args:
        frame: The state to draw, e.g. ``Frame(n=42, total=100, desc="demo")``.
        width: Line width in terminal cells.
        style: Glyphs, palette, and color depth (defaults to Braille truecolor).
        html: Return a notebook-styled HTML ``<pre>`` block instead of ANSI text.

    Returns:
        ANSI-styled text, one line per row of the bar, or the HTML block.

    Raises:
        ValueError: If ``width`` is less than 1.
    """
    if width < 1:
        raise ValueError("width must be at least 1")
    style = Style() if style is None else style
    rows = layout(frame, width, style)
    if html:
        return html_block(html_rows([to_html(row, style.depth) for row in rows]))
    return "\n".join(to_ansi(row, style.depth) for row in rows)
