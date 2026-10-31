"""Tests for the ``psnek`` command line."""

import importlib.metadata
import io
import logging
import os
import subprocess
import sys
import time
import types
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import cast

import pytest

import psnek
from psnek import __main__ as cli
from psnek.__main__ import main
from psnek._pace import STALL_AFTER
from psnek._render import GALLERY
from tests.fakes import FailingBytes, FailingStream

Wire = Callable[[io.BytesIO], tuple[io.BytesIO, io.StringIO]]
POSIX_ONLY = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX pipes and resource limits"
)


def _missing_metadata(name: str) -> str:
    raise importlib.metadata.PackageNotFoundError(name)


class _Interrupting(io.BytesIO):
    """A byte source that raises a configured exception on the first read."""

    def __init__(self, error: type[BaseException]) -> None:
        super().__init__()
        self.error = error

    def readinto1(self, buffer: object) -> int:
        raise self.error


class _Trickle(io.BytesIO):
    """A byte source that hands out one line per read, like a live pipe."""

    def __init__(self, lines: list[bytes]) -> None:
        super().__init__()
        self.lines = lines
        self.copied: list[bytes] = []
        self.sink = io.BytesIO()

    def readinto1(self, buffer: object) -> int:
        self.copied.append(self.sink.getvalue())
        if not self.lines:
            return 0
        line = self.lines.pop(0)
        cast("bytearray", buffer)[: len(line)] = line
        return len(line)


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> Iterator[Wire]:
    """Point stdin/stdout at in-memory byte streams and stderr at text."""
    with ExitStack() as stack:

        def connect(stdin: io.BytesIO) -> tuple[io.BytesIO, io.StringIO]:
            stdout = io.BytesIO()
            stderr = io.StringIO()
            reader = stack.enter_context(io.TextIOWrapper(stdin, encoding="utf-8"))
            writer = stack.enter_context(io.TextIOWrapper(stdout, encoding="utf-8"))
            monkeypatch.setattr(sys, "stdin", reader)
            monkeypatch.setattr(sys, "stdout", writer)
            monkeypatch.setattr(sys, "stderr", stderr)
            return stdout, stderr

        yield connect
        monkeypatch.undo()


def test_pipe_copies_and_counts_bytes(wire: Wire) -> None:
    # Arrange
    payload = b"line one\nline two\n\x00\xff binary tail"
    stdout, stderr = wire(io.BytesIO(payload))

    # Act
    code = main(["pipe", "--desc", "copy", "--total", "5"])

    # Assert
    assert code == 0
    assert stdout.getvalue() == payload
    assert "copy:" in stderr.getvalue()
    assert "3/3" in stderr.getvalue()
    assert "done" in stderr.getvalue()
    _, byte_stderr = wire(io.BytesIO(b"x" * 2048))
    assert main(["pipe", "--bytes", "--total", "2048"]) == 0
    assert "2.00KiB/2.00KiB" in byte_stderr.getvalue()


def test_pipe_passes_data_through_as_it_arrives(wire: Wire) -> None:
    # Arrange
    source = _Trickle([b"first\n", b"second\n"])
    source.sink, _ = wire(source)

    # Act
    code = main(["pipe"])

    # Assert
    assert code == 0
    assert source.copied == [b"", b"first\n", b"first\nsecond\n"]


def test_pipe_read_failures_end_the_pipe(
    wire: Wire, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    calls: list[bool] = []
    monkeypatch.setattr(cli, "_silence_stdout", lambda: calls.append(True))
    cases: list[tuple[type[BaseException], int, str, str]] = [
        (KeyboardInterrupt, 130, "", ""),
        (BrokenPipeError, 1, "output closed early", "interrupted"),
        (OSError, 1, "psnek pipe failed", "failed"),
    ]

    for error, expected, logged, shown in cases:
        caplog.clear()
        _, stderr = wire(_Interrupting(error))

        # Act
        code = main(["pipe"])

        # Assert
        assert code == expected
        assert logged in caplog.text
        assert shown in stderr.getvalue()
    assert calls == []


def test_pipe_write_failures_end_the_pipe(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    cases: list[tuple[OSError, str, str]] = [
        (BrokenPipeError("reader went away"), "output closed early", "interrupted"),
        (OSError("disk full"), "psnek pipe failed: OSError: disk full", "failed"),
    ]

    for error, logged, shown in cases:
        stderr = io.StringIO()
        stdin = types.SimpleNamespace(buffer=io.BytesIO(b"a\n"))
        monkeypatch.setattr(sys, "stdin", stdin)
        stdout = types.SimpleNamespace(buffer=FailingBytes(error))
        monkeypatch.setattr(sys, "stdout", stdout)
        monkeypatch.setattr(sys, "stderr", stderr)

        # Act
        code = main(["pipe"])

        # Assert
        assert code == 1
        assert logged in caplog.text
        assert shown in stderr.getvalue()


def test_demo_reports_print_failures(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    cases: list[tuple[io.StringIO | None, int, str]] = [
        (FailingStream(OSError("disk full")), 1, "OSError: disk full"),
        (None, 1, "could not print the gallery: OSError: stdout is closed"),
        (FailingStream(KeyboardInterrupt()), 130, "interrupted"),
        (FailingStream(BrokenPipeError("gone")), 1, "output closed early"),
    ]

    for stdout, expected, logged in cases:
        caplog.clear()
        monkeypatch.setattr(sys, "stdout", stdout)

        # Act
        code = main(["demo", "--width", "40"])

        # Assert
        assert code == expected
        assert logged in caplog.text
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(cli, "play", _interrupt_play)
    assert main(["demo", "--width", "40", "--play", "1"]) == 130


def _interrupt_play(seconds: float, height: int = 1) -> None:
    raise KeyboardInterrupt


def test_silencing_a_stdout_without_a_descriptor_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    for stdout in (None, FailingStream(OSError("disk full"))):
        # Arrange
        caplog.clear()
        monkeypatch.setattr(sys, "stdout", stdout)

        # Act
        with caplog.at_level(logging.DEBUG, logger="psnek"):
            cli._silence_stdout()

        # Assert
        assert "could not silence stdout" in caplog.text


def test_the_console_entry_silences_stdout_only_after_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    cases = [(0, io.StringIO(), []), (1, io.StringIO(), [True]), (0, None, [])]
    cases.append((130, io.StringIO(), [True]))

    for code, stdout, silenced in cases:
        calls: list[bool] = []
        monkeypatch.setattr(sys, "stdout", stdout)
        monkeypatch.setattr(cli, "main", lambda code=code: code)
        monkeypatch.setattr(
            cli, "_silence_stdout", lambda calls=calls: calls.append(True)
        )

        # Act
        with pytest.raises(SystemExit) as exit_info:
            cli.run()

        # Assert
        assert exit_info.value.code == code
        assert calls == silenced


class _BrokenFlush(io.StringIO):
    def flush(self) -> None:
        raise BrokenPipeError("reader went away")


def _leave(code: int | None) -> Callable[[], int]:
    def leave() -> int:
        raise SystemExit(code)

    return leave


def test_the_console_entry_fails_when_stdout_cannot_be_flushed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setattr(sys, "argv", ["psnek", "--help"])
    entries = [main, _leave(0), _leave(None)]

    for entry in entries:
        calls: list[bool] = []
        monkeypatch.setattr(sys, "stdout", _BrokenFlush())
        monkeypatch.setattr(cli, "main", entry)
        monkeypatch.setattr(
            cli, "_silence_stdout", lambda calls=calls: calls.append(True)
        )

        # Act
        with pytest.raises(SystemExit) as exit_info:
            cli.run()

        # Assert
        assert exit_info.value.code == 1
        assert calls == [True]


@pytest.mark.integration
def test_silencing_stdout_points_its_descriptor_at_the_null_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    read_end, write_end = os.pipe()
    with open(read_end, "rb") as reader, open(write_end, "wb", buffering=0) as writer:
        monkeypatch.setattr(sys, "stdout", writer)

        # Act
        cli._silence_stdout()
        writer.write(b"discarded")
        writer.close()
        received = reader.read()

    # Assert
    assert received == b""


class _WouldBlockOnce(io.BytesIO):
    """A byte source that has nothing ready once, like a non-blocking pipe."""

    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.waited = False

    def readinto1(self, buffer: object) -> int | None:  # type: ignore[override]
        if not self.waited:
            self.waited = True
            return None
        return super().readinto1(cast("bytearray", buffer))


class _FullOnce(io.BytesIO):
    """A byte sink that takes part of its first write and fails its first flush."""

    def __init__(self) -> None:
        super().__init__()
        self.full = self.unflushed = True

    def write(self, data: object) -> int:
        chunk = bytes(cast("memoryview", data))
        if self.full:
            self.full = False
            super().write(chunk[:2])
            raise BlockingIOError(11, "full", 2)
        return super().write(chunk)

    def flush(self) -> None:
        if self.unflushed:
            self.unflushed = False
            raise BlockingIOError(11, "full", 0)


def test_pipe_and_demo_wait_on_non_blocking_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    waits: list[bool] = []
    stdout = _FullOnce()
    monkeypatch.setattr(
        sys, "stdin", types.SimpleNamespace(buffer=_WouldBlockOnce(b"ab\ncd\n"))
    )
    monkeypatch.setattr(sys, "stdout", types.SimpleNamespace(buffer=stdout))
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    monkeypatch.setattr(cli, "_wait", lambda stream, writing: waits.append(writing))

    # Act
    code = main(["pipe"])

    # Assert
    assert code == 0
    assert stdout.getvalue() == b"ab\ncd\n"
    assert waits == [False, True, True]

    # Arrange
    blocking = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(blocking, encoding="utf-8"))
    main(["demo", "--width", "40", "--no-color"])
    sink = _FullOnce()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(sink, encoding="utf-8"))
    waits.clear()

    # Act
    code = main(["demo", "--width", "40", "--no-color"])

    # Assert
    assert code == 0
    assert sink.getvalue() == blocking.getvalue()
    assert waits == [True, True]


def test_pipe_needs_standard_streams(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    monkeypatch.setattr(sys, "stdin", io.StringIO())

    # Act
    code = main(["pipe"])

    # Assert
    assert code == 2
    assert "needs binary stdin and stdout" in caplog.text


def test_invalid_arguments_are_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    # Arrange
    invalid = [
        ["pipe", "--total", "-1"],
        ["pipe", "--total", "inf"],
        ["demo", "--width", "5"],
        ["demo", "--width", "wide"],
        ["demo", "--play", "0"],
        ["demo", "--phase", "nan"],
        ["demo", "--width", "1" + "0" * 400],
    ]

    for arguments in invalid:
        # Act
        with pytest.raises(SystemExit):
            main(arguments)

        # Assert
        assert "error:" in capsys.readouterr().err


def test_demo_prints_gallery_with_optional_color(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Act
    code = main(["demo", "--ascii", "--width", "60"])
    output = capsys.readouterr().out
    monkeypatch.setenv("FORCE_COLOR", "1")
    main(["demo", "--width", "80"])
    colored = capsys.readouterr().out
    main(["demo", "--width", "80", "--no-color"])
    plain = capsys.readouterr().out

    # Assert
    assert code == 0
    assert "coil" in output
    assert "\x1b" not in output
    assert max(map(len, output.splitlines())) <= 60
    assert "\x1b[38;5;" in colored
    assert "\x1b" not in plain


def test_demo_follows_earlier_text_and_falls_back_on_a_narrow_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    raw = io.BytesIO()
    with io.TextIOWrapper(raw, encoding="ascii") as stdout:
        monkeypatch.setattr(sys, "stdout", stdout)
        stdout.write("header\n")

        # Act
        code = main(["demo", "--width", "80"])
        stdout.flush()
        output = raw.getvalue().decode("ascii")

    # Assert
    assert code == 0
    assert output.startswith("header\n")
    assert output.count("header") == 1
    assert "#" in output
    assert "\\u65e5" in output


def test_demo_exports_html_and_plays(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    written: dict[str, str] = {}
    played: list[tuple[float, int]] = []

    def write_text(path: Path, text: str, encoding: str) -> int:
        written[str(path)] = text
        return len(text)

    def play(seconds: float, height: int) -> None:
        played.append((seconds, height))

    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(cli, "play", play)

    # Act
    code = main(["demo", "--html", "gallery.html", "--play", "2", "--height", "3"])

    # Assert
    page = written["gallery.html"]
    assert code == 0
    assert page.startswith("<!doctype html>")
    assert page.count("<pre") == 2
    assert "color:#" in page
    assert page.count("<br>") >= 2 * len(GALLERY)
    assert played == [(2.0, 3)]


def test_demo_reports_an_unwritable_html_path(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Arrange
    cases: list[tuple[str, OSError]] = [
        ("gallery.html", PermissionError("read-only")),
        ("", IsADirectoryError("is a directory")),
    ]

    for path, error in cases:

        def refuse(
            target: Path, text: str, encoding: str, error: OSError = error
        ) -> int:
            raise error

        monkeypatch.setattr(Path, "write_text", refuse)

        # Act
        code = main(["demo", "--html", path])

        # Assert
        assert code == 1
        assert f"could not write {path}: {type(error).__name__}: {error}" in caplog.text


def test_play_simulation_runs_and_long_plays_stall(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    monkeypatch.setenv("PSNEK_MODE", "log")
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    # Act
    cli.play(2.0)
    short_longest = max(sleeps)
    sleeps.clear()
    cli.play(20.0)

    # Assert
    errors = capsys.readouterr().err
    assert "epochs: 100.0% 3/3" in errors
    assert "streaming: 1020" in errors
    assert short_longest <= STALL_AFTER < max(sleeps)


def test_version_is_reported_when_metadata_is_installed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Act
    with pytest.raises(SystemExit) as installed:
        main(["--version"])
    reported = capsys.readouterr().out
    expected = f"psnek {psnek.__version__}\n"
    monkeypatch.setattr(importlib.metadata, "version", _missing_metadata)
    with pytest.raises(SystemExit) as missing:
        main(["--version"])

    # Assert
    assert installed.value.code == 0
    assert reported == expected
    assert missing.value.code == 1
    assert "version is unknown" in caplog.text


def test_commands_work_without_installed_metadata(
    wire: Wire, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(importlib.metadata, "version", _missing_metadata)
    wire(io.BytesIO(b"a\n"))

    # Act
    code = main(["pipe"])

    # Assert
    assert code == 0
    assert not hasattr(psnek, "__version__")
    with pytest.raises(AttributeError, match="no attribute"):
        _ = psnek.no_such_thing


@pytest.mark.integration
@POSIX_ONLY
def test_wait_returns_once_a_pipe_is_ready() -> None:
    # Arrange
    read_fd, write_fd = os.pipe()
    with open(read_fd, "rb", buffering=0) as reader, open(write_fd, "wb") as writer:
        writer.write(b"x")
        writer.flush()

        # Act
        cli._wait(cast("io.IOBase", writer), writing=True)
        cli._wait(cast("io.IOBase", reader), writing=False)

        # Assert
        assert reader.read(1) == b"x"


def _no_file_growth() -> None:
    import resource

    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))


@pytest.mark.integration
@POSIX_ONLY
def test_help_exits_cleanly_when_stdout_is_closed() -> None:
    # Arrange
    read_end, write_end = os.pipe()
    os.close(read_end)

    # Act
    with open(write_end, "wb") as stdout:
        result = subprocess.run(
            [sys.executable, "-m", "psnek", "--help"],
            stdout=stdout,
            stderr=subprocess.PIPE,
            check=False,
        )

    # Assert
    assert result.returncode == 1
    assert result.stderr == b""


@pytest.mark.integration
@POSIX_ONLY
def test_pipe_exits_cleanly_when_stdout_cannot_be_written(tmp_path: Path) -> None:
    # Arrange
    environment = {**os.environ, "PSNEK_MODE": "log"}

    # Act
    with (tmp_path / "out.txt").open("wb") as stdout:
        result = subprocess.run(
            [sys.executable, "-m", "psnek", "pipe"],
            input=b"a\n",
            stdout=stdout,
            stderr=subprocess.PIPE,
            env=environment,
            preexec_fn=_no_file_growth,
            check=False,
        )

    # Assert
    assert result.returncode == 1
    assert b"psnek pipe failed" in result.stderr
    assert b"Exception ignored" not in result.stderr
