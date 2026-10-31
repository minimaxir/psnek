"""Tests for environment detection."""

import ctypes
import io
import logging
import os
import sys
import types
from collections.abc import Iterator
from typing import TextIO, cast

import pytest

from psnek import _deferred, _env
from psnek._env import (
    color_depth,
    detect_mode,
    enable_windows_vt,
    env_flag,
    env_float,
    glyphs_for,
    html_color_depth,
    in_notebook,
    is_tty,
    notification_protocol,
    supports_progress_osc,
    terminal_size,
)
from psnek._glyphs import ASCII, BRAILLE
from psnek._style import ColorDepth
from tests.fakes import FakeTerminal


@pytest.fixture(autouse=True)
def flush_deferred() -> Iterator[None]:
    """Emit diagnostics queued by the detection functions after each test."""
    yield
    _deferred.run_deferred()


def _install_shell(monkeypatch: pytest.MonkeyPatch, class_name: str) -> None:
    shell = type(class_name, (), {})()
    module = types.ModuleType("IPython")
    monkeypatch.setattr(module, "get_ipython", lambda: shell, raising=False)
    monkeypatch.setitem(sys.modules, "IPython", module)


def test_mode_defaults_to_live_only_on_an_interactive_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    automation = [(name, "1") for name in _env.CI_VARIABLES + _env.AGENT_VARIABLES]
    automation.append(("TERM", "dumb"))

    # Act
    defaults = (detect_mode(FakeTerminal()), detect_mode(io.StringIO()))
    automated = []
    for name, value in automation:
        with monkeypatch.context() as patch:
            patch.setenv(name, value)
            automated.append(detect_mode(FakeTerminal()))

    # Assert
    assert defaults == ("live", "log")
    assert set(automated) == {"log"}


def test_forced_mode_takes_precedence_and_unknown_modes_warn_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    monkeypatch.setenv("CI", "1")
    monkeypatch.setenv("PSNEK_MODE", "live")

    # Act
    forced = detect_mode(io.StringIO())
    monkeypatch.setenv("PSNEK_MODE", "bogus")
    ignored = [detect_mode(FakeTerminal()) for _ in range(3)]
    _deferred.run_deferred()

    # Assert
    assert forced == "live"
    assert ignored == ["log"] * 3
    assert caplog.text.count("ignoring unknown PSNEK_MODE='bogus'") == 1


def test_forced_live_falls_back_without_escape_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("PSNEK_MODE", "live")
    monkeypatch.setattr(_env, "enable_windows_vt", lambda stream: False)

    # Act
    mode = detect_mode(FakeTerminal())

    # Assert
    assert mode == "log"


@pytest.mark.parametrize(
    ("shell", "stream", "expected"),
    [
        ("ZMQInteractiveShell", sys.stderr, "notebook"),
        ("TerminalInteractiveShell", FakeTerminal(), "log"),
    ],
)
def test_notebook_detection(
    monkeypatch: pytest.MonkeyPatch, shell: str, stream: TextIO, expected: str
) -> None:
    # Arrange
    _install_shell(monkeypatch, shell)
    if expected == "log":
        monkeypatch.setenv("PSNEK_MODE", "notebook")

    # Act
    detected = in_notebook()
    mode = detect_mode(stream)

    # Assert
    assert detected is (expected == "notebook")
    assert mode == expected


def test_workers_forked_from_a_kernel_hide_their_bars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    _install_shell(monkeypatch, "ZMQInteractiveShell")
    worker_stream = types.SimpleNamespace(_is_master_process=lambda: False)
    monkeypatch.setattr(sys, "stderr", worker_stream)

    # Act
    modes = []
    for forced in ("", "notebook"):
        monkeypatch.setenv("PSNEK_MODE", forced)
        modes.append(detect_mode(sys.stderr))

    # Assert
    assert modes == ["off", "off"]


def test_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    cases = [("", False), ("0", False), ("false", False), ("Off", False), ("1", True)]

    for value, expected in cases:
        # Act
        monkeypatch.setenv("PSNEK_ASCII", value)

        # Assert
        assert env_flag("PSNEK_ASCII") is expected


def test_color_depth_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    cases: list[tuple[dict[str, str], bool, ColorDepth]] = [
        ({}, True, ColorDepth.EIGHT_BIT),
        ({}, False, ColorDepth.NONE),
        ({"TERM": "xterm-kitty"}, True, ColorDepth.TRUE_COLOR),
        ({"COLORTERM": "24BIT"}, True, ColorDepth.TRUE_COLOR),
        ({"WT_SESSION": "1"}, True, ColorDepth.TRUE_COLOR),
        ({"TERM_PROGRAM": "iTerm.app"}, True, ColorDepth.TRUE_COLOR),
        ({"FORCE_COLOR": "1"}, False, ColorDepth.EIGHT_BIT),
        ({"FORCE_COLOR": "3"}, False, ColorDepth.TRUE_COLOR),
        ({"NO_COLOR": "1", "FORCE_COLOR": "3"}, True, ColorDepth.NONE),
    ]

    for variables, interactive, expected in cases:
        with monkeypatch.context() as patch:
            for name, value in variables.items():
                patch.setenv(name, value)
            stream = FakeTerminal() if interactive else io.StringIO()

            # Act
            depth = color_depth(stream)

            # Assert
            assert depth is expected, variables
    monkeypatch.setenv("NO_COLOR", "1")
    assert html_color_depth() is ColorDepth.NONE


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ({}, False),
        ({"WT_SESSION": "x"}, True),
        ({"TERM_PROGRAM": "ghostty"}, True),
        ({"TERM_PROGRAM": "iTerm.app", "TERM_PROGRAM_VERSION": "3.6.6"}, True),
        ({"TERM_PROGRAM": "iTerm.app", "TERM_PROGRAM_VERSION": "3.6.6beta1"}, False),
        ({"TERM_PROGRAM": "iTerm.app"}, False),
        ({"TERM_PROGRAM": "ghostty", "TMUX": "/tmp/tmux"}, False),
        ({"TERM_PROGRAM": "ghostty", "STY": "1.pts"}, False),
        ({"TERM_PROGRAM": "ghostty", "TERM": "screen-256color"}, False),
        ({"PSNEK_OSC_PROGRESS": "1"}, True),
        ({"TERM_PROGRAM": "ghostty", "PSNEK_OSC_PROGRESS": "0"}, False),
    ],
)
def test_progress_osc_allowlist(
    monkeypatch: pytest.MonkeyPatch, variables: dict[str, str], expected: bool
) -> None:
    # Arrange
    for name, value in variables.items():
        monkeypatch.setenv(name, value)

    # Act / Assert
    assert supports_progress_osc(FakeTerminal()) is expected
    assert supports_progress_osc(io.StringIO()) is False


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ({}, None),
        ({"TERM_PROGRAM": "iTerm.app"}, "osc9"),
        ({"TERM_PROGRAM": "ghostty"}, "osc777"),
        ({"TERM_PROGRAM": "WezTerm"}, "osc777"),
        ({"TERM": "foot-extra"}, "osc777"),
        ({"TERM": "xterm-kitty"}, "osc99"),
        ({"KITTY_WINDOW_ID": "1"}, "osc99"),
        ({"TERM_PROGRAM": "vscode"}, None),
        ({"TERM_PROGRAM": "ghostty", "TMUX": "/tmp/tmux"}, None),
        ({"TERM": "xterm-kitty", "STY": "1.pts"}, None),
        ({"PSNEK_NOTIFY": "1"}, "osc9"),
        ({"PSNEK_NOTIFY": "on", "TERM": "xterm-kitty"}, "osc99"),
        ({"PSNEK_NOTIFY": "bell", "TMUX": "/tmp/tmux"}, "bell"),
        ({"PSNEK_NOTIFY": "OSC777"}, "osc777"),
        ({"PSNEK_NOTIFY": "0", "TERM_PROGRAM": "ghostty"}, None),
    ],
)
def test_notification_protocol_allowlist(
    monkeypatch: pytest.MonkeyPatch, variables: dict[str, str], expected: str | None
) -> None:
    # Arrange
    for name, value in variables.items():
        monkeypatch.setenv(name, value)

    # Act / Assert
    assert notification_protocol(FakeTerminal()) == expected
    assert notification_protocol(io.StringIO()) is None


def test_unknown_notification_setting_is_ignored_with_one_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    monkeypatch.setenv("TERM_PROGRAM", "ghostty")
    monkeypatch.setenv("PSNEK_NOTIFY", "growl")

    # Act
    protocols = [notification_protocol(FakeTerminal()) for _ in range(3)]
    _deferred.run_deferred()

    # Assert
    assert protocols == ["osc777"] * 3
    assert caplog.text.count("ignoring unknown PSNEK_NOTIFY='growl'") == 1


def test_glyphs_follow_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    with io.TextIOWrapper(io.BytesIO(), encoding="latin-1") as latin:
        assert glyphs_for(latin) is ASCII
    assert glyphs_for(io.StringIO()) is BRAILLE
    monkeypatch.setenv("PSNEK_ASCII", "1")
    assert glyphs_for(io.StringIO()) is ASCII


def test_a_stream_that_cannot_answer_isatty_is_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    closed = io.StringIO()
    closed.close()

    # Act
    with caplog.at_level(logging.DEBUG, logger="psnek"):
        answers = [is_tty(closed) for _ in range(3)]
        _deferred.run_deferred()

    # Assert
    assert answers == [False] * 3
    assert caplog.text.count("treating stream as non-interactive") == 1


def test_terminal_size_falls_back_to_bounded_environment(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    monkeypatch.setenv("COLUMNS", "999999")
    monkeypatch.setenv("LINES", "x")
    _env.once.cache_clear()

    # Act
    with caplog.at_level(logging.DEBUG, logger="psnek"):
        sizes = {terminal_size(io.StringIO()) for _ in range(3)}
        _deferred.run_deferred()

    # Assert
    assert sizes == {(1000, 24)}
    assert caplog.text.count("using COLUMNS/LINES") == 1


def test_a_terminal_reporting_zero_size_uses_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.delenv("LINES", raising=False)
    monkeypatch.setattr(os, "get_terminal_size", lambda fd: os.terminal_size((0, 0)))

    # Act
    size = terminal_size(cast("TextIO", types.SimpleNamespace(fileno=lambda: 1)))

    # Assert
    assert size == (120, 24)


def test_version_parsing() -> None:
    # Arrange
    ordered = ["", "0.1", "3.6.5", "3.6.6beta1", "3.6.6", "3.6.10", "3.7beta1", "3.7"]

    # Act
    versions = [_env._version(text) for text in ordered]

    # Assert
    assert versions == sorted(versions)
    assert len(set(versions)) == len(versions)
    assert _env._version("3." + "9" * 5000) == (3, 999_999_999)
    for text in ("3.6.6 ", "3.6.6+build", "3.6.6-1"):
        assert _env._version(text) == (3, 6, 6)


@pytest.mark.parametrize(
    ("raw", "expected", "warning"),
    [
        ("0", 0.1, "using PSNEK_LOG_INTERVAL=0.1"),
        ("30s", 10.0, "PSNEK_LOG_INTERVAL"),
        ("  ", 10.0, None),
    ],
)
def test_env_floats_are_validated_with_one_warning(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    raw: str,
    expected: float,
    warning: str | None,
) -> None:
    # Arrange
    monkeypatch.setenv("PSNEK_LOG_INTERVAL", raw)
    _env.once.cache_clear()

    # Act
    values = {env_float("PSNEK_LOG_INTERVAL", 10.0, 0.1, 60.0) for _ in range(3)}
    _deferred.run_deferred()

    # Assert
    assert values == {expected}
    if warning is None:
        assert "PSNEK_LOG_INTERVAL" not in caplog.text
    else:
        assert caplog.text.count(warning) == 1


class _FakeKernel32:
    """Console mode calls answering as configured, recording the mode set."""

    def __init__(self, is_console: bool, accepts: bool) -> None:
        self.is_console = is_console
        self.accepts = accepts
        self.set_modes: list[int] = []

    def GetConsoleMode(self, handle: int, mode: object) -> int:  # noqa: N802
        return int(self.is_console)

    def SetConsoleMode(self, handle: int, flags: int) -> int:  # noqa: N802
        self.set_modes.append(flags)
        return int(self.accepts)


@pytest.mark.parametrize(
    ("has_handle", "is_console", "accepts", "enabled"),
    [
        (False, True, True, False),
        (True, True, False, False),
        (True, True, True, True),
    ],
)
def test_windows_consoles_get_escape_processing_when_they_accept_it(
    monkeypatch: pytest.MonkeyPatch,
    has_handle: bool,
    is_console: bool,
    accepts: bool,
    enabled: bool,
) -> None:
    # Arrange
    kernel32 = _FakeKernel32(is_console, accepts)

    def get_osfhandle(descriptor: int) -> int:
        if not has_handle:
            raise OSError("bad descriptor")
        return 7

    msvcrt = types.ModuleType("msvcrt")
    monkeypatch.setattr(msvcrt, "get_osfhandle", get_osfhandle, raising=False)
    monkeypatch.setitem(sys.modules, "msvcrt", msvcrt)
    monkeypatch.setattr(
        ctypes, "WinDLL", lambda *args, **kwargs: kernel32, raising=False
    )
    monkeypatch.setattr(sys, "platform", "win32")
    stream = types.SimpleNamespace(fileno=lambda: 2)

    # Act
    result = enable_windows_vt(cast("TextIO", stream))
    _deferred.run_deferred()

    # Assert
    assert result is enabled
    assert kernel32.set_modes == ([0x4] if has_handle and is_console else [])
