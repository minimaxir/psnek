"""Command line interface: ``psnek pipe`` and ``psnek demo``."""

import argparse
import importlib.metadata
import io
import logging
import math
import os
import select
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar, cast

from psnek._bar import Bar, Options
from psnek._deferred import describe, run_deferred
from psnek._env import (
    color_depth,
    env_flag,
    glyphs_for,
    html_color_depth,
    terminal_size,
)
from psnek._glyphs import ASCII, BRAILLE, Style
from psnek._pace import STALL_AFTER
from psnek._render import gallery
from psnek._style import ColorDepth
from psnek._text import encodable
from psnek._track import track

logger = logging.getLogger("psnek")

N = TypeVar("N", int, float)

CHUNK_SIZE, MIN_WIDTH, MAX_WIDTH, MAX_HEIGHT = 1 << 16, 20, 1000, 8
MAX_PLAY, MAX_PHASE, STALL = 3600.0, 1e6, STALL_AFTER + 2
PAGE_STYLE = "margin:0;font-family:system-ui,sans-serif"
_DESCRIPTION = (
    "psnek pipe copies stdin to stdout while showing progress on stderr, and "
    "psnek demo renders the design gallery or plays a simulated live session."
)


def _bounded(kind: Callable[[str], N], low: N, high: float) -> Callable[[str], N]:
    """Build an argparse type accepting finite ``kind`` values in ``[low, high]``."""

    def parse(text: str) -> N:
        try:
            value = kind(text)
            valid = math.isfinite(value) and low <= value <= high
        except (ValueError, OverflowError) as error:
            raise argparse.ArgumentTypeError(str(error)) from error
        if not valid:
            limit = f"at least {low}" if high == math.inf else f"from {low} to {high}"
            raise argparse.ArgumentTypeError(f"must be a finite number {limit}")
        return value

    return parse


class _Version(argparse.Action):
    def __call__(self, parser: argparse.ArgumentParser, *args: object) -> None:
        """Print ``psnek <version>`` and exit through ``parser``; 1 if not installed."""
        try:
            version = importlib.metadata.version("psnek")
        except importlib.metadata.PackageNotFoundError:
            logger.error("psnek is not installed, so its version is unknown")
            parser.exit(1)
        print(f"psnek {version}")
        parser.exit()


def _exit_code(action: Callable[[], object], failure: str) -> int:
    """Run ``action``, logging a failure and returning the process exit code."""
    try:
        action()
    except BrokenPipeError:
        logger.warning("output closed early; stopping")
        return 1
    except OSError as error:
        logger.error(f"{failure}: {describe(error)}")
        return 1
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130
    return 0


def _silence_stdout() -> None:
    try:
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
    except (AttributeError, OSError, ValueError) as error:
        logger.debug(f"could not silence stdout: {describe(error)}")


def _flush_stdout() -> bool:
    stdout = sys.stdout
    if stdout is None:
        return True
    try:
        _drain(stdout.flush, cast("io.IOBase", getattr(stdout, "buffer", stdout)))
    except (AttributeError, OSError, ValueError) as error:
        logger.debug(f"could not flush stdout: {describe(error)}")
        return False
    return True


def _wait(stream: io.IOBase, writing: bool) -> None:
    readers, writers = ([], [stream]) if writing else ([stream], [])
    select.select(readers, writers, [])


def _read(reader: io.BufferedReader, buffer: bytearray) -> int:
    """Read what is available into ``buffer``; 0 only at the end of the input."""
    while (size := reader.readinto1(buffer)) is None:
        _wait(reader, writing=False)
    return size


def _print(text: str) -> None:
    """Print ``text`` and a newline, waiting whenever a non-blocking stdout is full."""
    if sys.stdout is None:
        raise OSError("stdout is closed")
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    text = encodable(text, encoding) + "\n"
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is None:
        print(text, end="", flush=True)
        return
    _drain(sys.stdout.flush, buffer)
    _send(buffer, memoryview(text.encode(encoding)))


def _send(writer: io.BufferedWriter, data: memoryview) -> None:
    while data:
        try:
            data = data[writer.write(data) :]
        except BlockingIOError as error:
            data = data[error.characters_written :]
            _wait(writer, writing=True)
    _drain(writer.flush, writer)


def _drain(flush: Callable[[], None], writer: io.IOBase) -> None:
    while True:
        try:
            return flush()
        except BlockingIOError:
            _wait(writer, writing=True)


def pipe(total: float | None, desc: str, count_bytes: bool) -> int:
    """Copy stdin to stdout, counting lines or bytes; return the exit code."""
    reader = cast("io.BufferedReader | None", getattr(sys.stdin, "buffer", None))
    writer = cast("io.BufferedWriter | None", getattr(sys.stdout, "buffer", None))
    if reader is None or writer is None:
        logger.error("psnek pipe needs binary stdin and stdout")
        return 2
    options = Options(
        unit="B" if count_bytes else "line",
        scale="iec" if count_bytes else "none",
        capture=False,
    )

    def copy() -> None:
        buffer = bytearray(CHUNK_SIZE)
        view, open_line = memoryview(buffer), False
        with Bar(total, desc, options=options) as bar:
            try:
                while size := _read(reader, buffer):
                    _send(writer, view[:size])
                    bar.update(size if count_bytes else buffer.count(b"\n", 0, size))
                    open_line = buffer[size - 1] != ord("\n")
            except BrokenPipeError:
                bar.interrupt()
                raise
            if open_line and not count_bytes:
                bar.update()
            if bar.total is not None and bar.n < bar.total:
                bar.total = bar.n

    return _exit_code(copy, "psnek pipe failed")


def play(seconds: float, height: int = 1) -> None:
    """Run a deterministic simulated workload, stalling from twice `STALL` seconds."""
    options = Options(height=height)
    epochs, steps, chunks = 3, 40, 60
    stall = STALL if seconds >= 2 * STALL else 0.0
    pause = (seconds - stall) / (epochs * steps * 1.5 + chunks)
    history: list[dict[str, float]] = []
    outer = track(epochs, "epochs", options=options)
    for epoch in outer:
        bar = track(steps, f"epoch {epoch + 1}", options=options)
        loss = 1.0
        for step in bar:
            time.sleep(pause * (1 + math.sin(step * 0.7) ** 2))
            loss = 1 / (1 + epoch * steps + step)
            bar.status(loss=loss, lr=3e-4)
            if step % 9 == 0:
                bar.log(f"checkpoint written at step {epoch * steps + step}")
            if epoch == 1 and step == steps // 2:
                bar.panel({"shard": 3, "state": "waiting", "retries": 2})
                time.sleep(stall)
                outer.panel(history)
        history.append({"epoch": epoch + 1, "loss": loss, "seconds": bar.elapsed})
        outer.panel(history)
    with Bar(desc="streaming", options=options) as stream:
        for _ in range(chunks):
            time.sleep(pause)
            stream.update(17)


def demo(args: argparse.Namespace) -> int:
    """Print or export the gallery, and play a simulation, per ``args``."""
    terminal = terminal_size(sys.stdout)[0] - 1
    width = args.width or min(max(terminal, MIN_WIDTH), MAX_WIDTH)
    if args.html is None:
        glyphs = ASCII if args.ascii else glyphs_for(sys.stdout)
        depth = ColorDepth.NONE if args.no_color else color_depth(sys.stdout)
    else:
        glyphs = ASCII if args.ascii or env_flag("PSNEK_ASCII") else BRAILLE
        depth = ColorDepth.NONE if args.no_color else html_color_depth()
    style = Style(glyphs, depth=depth)
    body = gallery(width, style, args.html is not None, args.phase, args.height)
    if args.html is None:
        code = _exit_code(lambda: _print(body), "could not print the gallery")
    else:
        page = (
            '<!doctype html><meta charset="utf-8"><title>psnek gallery</title>'
            f'<body style="{PAGE_STYLE}">'
            f'<div style="padding:1em;background:#0d1117;color:#e6edf3">{body}</div>'
            f'<div style="padding:1em;background:#fff;color:#1f2328">{body}</div>'
            "</body>"
        )
        path = Path(args.html)
        code = _exit_code(
            lambda: path.write_text(page, encoding="utf-8"),
            f"could not write {args.html}",
        )
    if code or not args.play:
        return code
    return _exit_code(lambda: play(args.play, args.height), "the simulation failed")


def main(argv: list[str] | None = None) -> int:
    """Run the ``psnek`` command line on ``argv`` and return its exit code."""
    parser = argparse.ArgumentParser(prog="psnek", description=_DESCRIPTION)
    parser.add_argument("--version", action=_Version, nargs=0, help="print the version")
    commands = parser.add_subparsers(dest="command", required=True)
    piping = commands.add_parser("pipe", help="show progress for data piped through")
    piping.add_argument(
        "--total", type=_bounded(float, 0.0, math.inf), help="expected lines (or bytes)"
    )
    piping.add_argument("--desc", default="", help="bar description")
    piping.add_argument("--bytes", action="store_true", help="count bytes, not lines")
    showing = commands.add_parser("demo", help="render the design gallery")
    showing.add_argument(
        "--width",
        type=_bounded(int, MIN_WIDTH, MAX_WIDTH),
        help="line width in cells (default: terminal width)",
    )
    phase, height = _bounded(float, 0.0, MAX_PHASE), _bounded(int, 1, MAX_HEIGHT)
    showing.add_argument(
        "--phase", type=phase, default=0.0, help="animation clock in seconds"
    )

    showing.add_argument("--height", type=height, default=1, help="rows each bar spans")

    showing.add_argument("--ascii", action="store_true", help="ASCII glyphs")
    showing.add_argument("--no-color", action="store_true", help="disable color")
    showing.add_argument("--html", metavar="PATH", help="write the gallery as HTML")
    showing.add_argument(
        "--play",
        type=_bounded(float, 0.1, MAX_PLAY),
        metavar="SECONDS",
        help="simulate a run",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "pipe":
            return pipe(args.total, args.desc, args.bytes)
        return demo(args)
    finally:
        run_deferred()


def run() -> None:
    """Run `main` as the ``psnek`` process, silencing a failed stdout, and exit."""
    code: int | str | None
    try:
        code = main()
    except SystemExit as error:  # --help, --version and usage errors
        code = error.code
    flushed = _flush_stdout()
    if code or not flushed:
        _silence_stdout()
    sys.exit(code or (0 if flushed else 1))


if __name__ == "__main__":
    run()
