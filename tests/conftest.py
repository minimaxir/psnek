"""Shared fixtures: a hermetic environment and fake terminals."""

import math
import os
import sys
import threading
import types
from collections.abc import Iterator
from time import monotonic, sleep

import pytest

from psnek import _backends, _bar, _deferred, _engine
from psnek._backends import NotebookBackend
from psnek._env import once
from tests.fakes import FakeEvents, FakeHandle, FakeTerminal

_ENV_PREFIXES = ("PSNEK_", "CLAUDE", "CODEX_", "CURSOR_", "COPILOT_")
_ENV_NAMES = {
    "NO_COLOR",
    "FORCE_COLOR",
    "COLORTERM",
    "TERM_PROGRAM",
    "TERM_PROGRAM_VERSION",
    "WT_SESSION",
    "CONEMUANSI",
    "TMUX",
    "STY",
    "TERM",
    "KITTY_WINDOW_ID",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--update-golden`` to rewrite snapshot files."""
    parser.addoption("--update-golden", action="store_true", help="rewrite goldens")


@pytest.fixture(autouse=True)
def hermetic_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove CI, agent, color, and psnek variables; fix the terminal size."""
    from psnek._env import AGENT_VARIABLES, CI_VARIABLES

    for name in list(os.environ):
        if name in _ENV_NAMES or name.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(name)
    for name in CI_VARIABLES + AGENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setenv("LINES", "24")
    yield
    try:
        assert not any(engine.sources for engine in _engine.ENGINES.values())
        assert not _engine.CAPTURING
        assert not _engine.CAPTURES
        assert not isinstance(sys.stdout, _engine.Capture)
        assert not isinstance(sys.stderr, _engine.Capture)
        assert _drained()
    finally:
        _reset_psnek()


def _drained(timeout: float = 1.0) -> bool:
    """Wait for a stopping render thread to run the records it still owes."""
    deadline = monotonic() + timeout
    while _deferred.pending() and monotonic() < deadline:
        sleep(0.005)
    return not _deferred.pending()


def _reset_psnek() -> None:
    """Clear global state, so one leaking test does not fail every later one."""
    for engine in _engine.ENGINES.values():
        engine.stop_event.set()
    for thread in threading.enumerate():
        if thread.name == "psnek-render":
            thread.join(timeout=5)
    _engine.ENGINES.clear()
    _engine.CAPTURING.clear()
    _engine._OUTPUTS.clear()
    _backends._RUNNING.clear()
    _backends._LATEST.append(None)
    _engine._restore_streams()
    _deferred._pending.clear()
    _bar._THREAD.__dict__.clear()
    once.cache_clear()


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> FakeTerminal:
    """A fake TTY with live mode forced on, sharing the standard streams' terminal.

    No stream has a file to compare, so stdout and stderr count as writing to
    the fake terminal, as they do to a real one.
    """
    monkeypatch.setenv("PSNEK_MODE", "live")
    monkeypatch.setattr(_engine, "_file_of", lambda stream: None)
    return FakeTerminal()


@pytest.fixture
def update_golden(request: pytest.FixtureRequest) -> bool:
    """Whether snapshot files should be rewritten instead of compared."""
    return bool(request.config.getoption("--update-golden"))


@pytest.fixture
def fake_ipython(monkeypatch: pytest.MonkeyPatch) -> list[FakeHandle]:
    """Install a stand-in ``IPython.display`` module recording display calls."""
    handles: list[FakeHandle] = []

    def display(obj: dict[str, str], *, raw: bool, display_id: bool) -> FakeHandle:
        handle = FakeHandle()
        handle.updates.append(obj)
        handles.append(handle)
        return handle

    module = types.ModuleType("IPython.display")
    monkeypatch.setattr(module, "display", display, raising=False)
    monkeypatch.setitem(sys.modules, "IPython.display", module)
    monkeypatch.setattr(_engine, "detect_mode", lambda stream: "notebook")
    return handles


@pytest.fixture
def cell_events(
    fake_ipython: list[FakeHandle], monkeypatch: pytest.MonkeyPatch
) -> FakeEvents:
    """Install a stand-in IPython shell, running a cell, that psnek hooks.

    It reports cell starts and ends when told to.
    """
    events = FakeEvents()
    shell = types.SimpleNamespace(events=events, execution_count=1)
    module = types.ModuleType("IPython")
    monkeypatch.setattr(module, "get_ipython", lambda: shell, raising=False)
    monkeypatch.setitem(sys.modules, "IPython", module)
    _engine._hook_cells()
    events.start_cell()
    return events


@pytest.fixture
def never_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    """End each notebook session at once, instead of on the next render tick."""
    monkeypatch.setattr(NotebookBackend, "busy", property(lambda self: False))


@pytest.fixture
def always_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave each notebook session's end to the next render tick."""
    monkeypatch.setattr(NotebookBackend, "busy", property(lambda self: True))


@pytest.fixture
def uncapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send every notebook update at once, however large."""
    monkeypatch.setattr(_backends, "NOTEBOOK_RATE", math.inf)
