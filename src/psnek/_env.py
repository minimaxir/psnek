"""Environment detection: output mode, color depth, glyphs, and terminal size."""

import logging
import math
import os
import re
import sys
from functools import lru_cache
from typing import Literal, TextIO, get_args

from psnek._deferred import defer, describe
from psnek._glyphs import ASCII, BRAILLE, Glyphs
from psnek._style import ColorDepth

Mode = Literal["live", "log", "notebook", "off"]
MODES: tuple[Mode, ...] = get_args(Mode)
Notify = Literal["osc9", "osc777", "osc99", "bell"]
NOTIFY_PROTOCOLS: tuple[Notify, ...] = get_args(Notify)

# fmt: off
CI_VARIABLES = (
    "CI", "CONTINUOUS_INTEGRATION", "TRAVIS", "APPVEYOR", "DRONE", "SEMAPHORE",
    "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE", "TF_BUILD", "JENKINS_URL",
    "CIRCLECI", "TEAMCITY_VERSION", "BITBUCKET_COMMIT", "CODEBUILD_BUILD_ARN",
)
AGENT_VARIABLES = (
    "AI_AGENT", "CLAUDECODE", "CODEX_SANDBOX", "CODEX_THREAD_ID", "CURSOR_AGENT",
    "GEMINI_CLI", "QWEN_CODE", "CLINE_ACTIVE", "ROO_CODE_TASK_ID",
    "AMP_CURRENT_THREAD_ID", "COPILOT_AGENT", "OPENCODE", "AUGMENT_AGENT",
    "GOOSE_TERMINAL", "PI_CODING_AGENT",
)
# fmt: on
_FALSE = frozenset(("", "0", "false", "no", "off"))
_TRUE = frozenset(("1", "true", "yes", "on"))
_TRUECOLOR_PROGRAMS = {"iterm.app", "wezterm", "ghostty", "vscode", "warpterminal"}
_TRUECOLOR_TERMS = ("xterm-kitty", "xterm-ghostty", "alacritty", "foot", "contour")
_PROGRESS_PROGRAMS = {"ghostty", "wezterm"}
_NOTIFY_PROGRAMS: dict[str, Notify] = {
    "iterm.app": "osc9",
    "ghostty": "osc777",
    "wezterm": "osc777",
}
_ITERM_PROGRESS_VERSION = (3, 6, 6)
_LEADING_DIGITS = re.compile(r"\d{1,9}")
_ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
_STREAM_ERRORS = (AttributeError, ValueError, OSError)


@lru_cache(maxsize=64)
def once(level: int, message: str) -> None:
    """Queue a ``message`` at ``level`` unless among the last 64 distinct ones."""
    defer(level, message)


def env_flag(name: str) -> bool:
    """Return whether variable ``name`` is set, other than to 0, false, no, or off."""
    return os.environ.get(name, "").strip().lower() not in _FALSE


def env_float(name: str, default: float, low: float, high: float) -> float:
    """Read variable ``name`` as a float clamped to ``[low, high]`` (or ``default``)."""
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if math.isnan(value):
        once(logging.WARNING, f"ignoring {name}={raw!r}: not a number")
        return default
    clamped = min(max(value, low), high)
    if clamped != value:
        once(logging.WARNING, f"using {name}={clamped:g}: {raw!r} is out of range")
    return clamped


def is_tty(stream: TextIO) -> bool:
    """Return whether ``stream`` reports being a TTY, never raising."""
    try:
        return stream.isatty()
    except _STREAM_ERRORS as error:
        once(logging.DEBUG, f"treating stream as non-interactive: {describe(error)}")
        return False


def ipython_shell() -> object:
    """Return the running IPython shell, or None, without importing IPython."""
    return getattr(sys.modules.get("IPython"), "get_ipython", lambda: None)()


def in_notebook() -> bool:
    """Return whether this runs in a ZMQ-based IPython kernel (Jupyter, Colab)."""
    shell_type = type(ipython_shell())
    return any(cls.__name__ == "ZMQInteractiveShell" for cls in shell_type.__mro__)


def detect_mode(stream: TextIO) -> Mode:
    """Choose the output mode for bars on ``stream``: notebook, log, or live."""
    forced = os.environ.get("PSNEK_MODE", "").strip().lower()
    if forced and forced not in MODES:
        once(logging.WARNING, f"ignoring unknown PSNEK_MODE={forced!r}")
    mode: Mode
    if forced == "notebook" and not in_notebook():
        mode = "log"
    elif forced in MODES:
        mode = forced
    elif stream in (sys.stdout, sys.stderr) and in_notebook():
        mode = "notebook"
    elif (
        any(map(env_flag, CI_VARIABLES + AGENT_VARIABLES))
        or os.environ.get("TERM") == "dumb"
        or not is_tty(stream)
    ):
        mode = "log"
    else:
        mode = "live"
    is_kernel = getattr(stream, "_is_master_process", None)
    if mode == "notebook" and callable(is_kernel) and not is_kernel():
        return "off"
    return "log" if mode == "live" and not enable_windows_vt(stream) else mode


def color_depth(stream: TextIO) -> ColorDepth:
    """Choose the ANSI color depth of ``stream``; ``NO_COLOR`` beats ``FORCE_COLOR``."""
    if os.environ.get("NO_COLOR"):
        return ColorDepth.NONE
    if not env_flag("FORCE_COLOR") and not is_tty(stream):
        return ColorDepth.NONE
    program = os.environ.get("TERM_PROGRAM", "").lower()
    if (
        os.environ.get("COLORTERM", "").lower() in {"truecolor", "24bit"}
        or os.environ.get("FORCE_COLOR") == "3"
        or "WT_SESSION" in os.environ
        or program in _TRUECOLOR_PROGRAMS
        or os.environ.get("TERM", "").startswith(_TRUECOLOR_TERMS)
    ):
        return ColorDepth.TRUE_COLOR
    return ColorDepth.EIGHT_BIT


def html_color_depth() -> ColorDepth:
    """Choose the HTML color depth: none under ``NO_COLOR``, else true color."""
    return ColorDepth.NONE if os.environ.get("NO_COLOR") else ColorDepth.TRUE_COLOR


def _version(text: str) -> tuple[int, ...]:
    """Parse a dotted version; a letter suffix sorts it just below its release.

    Example:
        >>> base = (3, 6, 6)
        >>> _version("3.6.6") == base, _version("3.6.6+build") == base
        (True, True)
        >>> _version("3.6.6beta1") < base < _version("3.7beta1") < (3, 7)
        True
    """
    parts: list[int] = []
    for piece in text.strip().split("."):
        match = _LEADING_DIGITS.match(piece)
        if match is None:
            break
        parts.append(int(match.group()))
        suffix = piece[match.end() :]
        if suffix[:1].isalpha():
            parts[-1] -= 1
            parts.append(sys.maxsize)
        if suffix:
            break
    return tuple(parts)


def supports_progress_osc(stream: TextIO) -> bool:
    """Return whether ``stream``'s terminal is known to render OSC 9;4 progress."""
    if not is_tty(stream):
        return False
    if os.environ.get("PSNEK_OSC_PROGRESS", "").strip():
        return env_flag("PSNEK_OSC_PROGRESS")
    if _multiplexed():
        return False
    program = os.environ.get("TERM_PROGRAM", "").lower()
    if program == "iterm.app":
        version = _version(os.environ.get("TERM_PROGRAM_VERSION", ""))
        return version >= _ITERM_PROGRESS_VERSION
    return (
        "WT_SESSION" in os.environ
        or os.environ.get("CONEMUANSI") == "ON"
        or program in _PROGRESS_PROGRAMS
    )


def _multiplexed() -> bool:
    """Report a tmux or screen session, which drops unknown OSC sequences."""
    term = os.environ.get("TERM", "")
    return "TMUX" in os.environ or "STY" in os.environ or term.startswith("screen")


def _notify_protocol() -> Notify | None:
    if _multiplexed():
        return None
    term = os.environ.get("TERM", "")
    if term == "xterm-kitty" or "KITTY_WINDOW_ID" in os.environ:
        return "osc99"
    if term.startswith("foot"):
        return "osc777"
    return _NOTIFY_PROGRAMS.get(os.environ.get("TERM_PROGRAM", "").lower())


def notification_protocol(stream: TextIO) -> Notify | None:
    """Return how ``stream``'s terminal shows notifications, per ``PSNEK_NOTIFY``."""
    if not is_tty(stream):
        return None
    setting = os.environ.get("PSNEK_NOTIFY", "").strip().lower()
    if setting in NOTIFY_PROTOCOLS:
        return setting
    if setting and setting in _FALSE:
        return None
    if setting in _TRUE:
        return _notify_protocol() or "osc9"
    if setting:
        once(logging.WARNING, f"ignoring unknown PSNEK_NOTIFY={setting!r}")
    return _notify_protocol()


def glyphs_for(stream: TextIO) -> Glyphs:
    """Pick Braille glyphs when ``stream`` can encode them, else ASCII."""
    if env_flag("PSNEK_ASCII"):
        return ASCII
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        "".join(BRAILLE.partials).encode(encoding)
    except (UnicodeEncodeError, LookupError) as error:
        once(logging.DEBUG, f"using ASCII glyphs for {encoding}: {describe(error)}")
        return ASCII
    return BRAILLE


def terminal_size(stream: TextIO) -> tuple[int, int]:
    """Return ``stream``'s ``(columns, rows)``, else ``COLUMNS``/``LINES`` or 80x24."""
    try:
        columns, rows = os.get_terminal_size(stream.fileno())
    except _STREAM_ERRORS as error:
        once(logging.DEBUG, f"using COLUMNS/LINES for the size: {describe(error)}")
        columns = rows = 0
    columns = columns or int(env_float("COLUMNS", 80, 20, 1000))
    rows = rows or int(env_float("LINES", 24, 2, 1000))
    return min(max(columns, 2), 1000), min(max(rows, 2), 1000)


def enable_windows_vt(stream: TextIO) -> bool:
    """Enable escape processing on a Windows console ``stream``; report success."""
    if sys.platform != "win32":
        return True
    import ctypes  # noqa: PLC0415
    import msvcrt  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    try:
        handle = msvcrt.get_osfhandle(stream.fileno())
    except _STREAM_ERRORS as error:
        once(logging.DEBUG, f"no console handle: {describe(error)}")
        return False
    mode = wintypes.DWORD()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        once(logging.DEBUG, "stream is not a console; using log mode")
        return False
    flags = mode.value | _ENABLE_VIRTUAL_TERMINAL_PROCESSING
    if kernel32.SetConsoleMode(handle, flags):
        return True
    once(logging.DEBUG, "console refused escape processing; using log mode")
    return False
