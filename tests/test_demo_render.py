"""Tests for ``demo/render.py``, which records tapes and crops them to their text."""

import doctest
import importlib.util
import logging
import subprocess
import types
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parent.parent / "demo"


def _load(name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, DEMO / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render = _load("render")


def _pgm(width: int, height: int, lit: set[tuple[int, int]]) -> bytes:
    pixels = bytes(
        200 if (x, y) in lit else 0 for y in range(height) for x in range(width)
    )
    return f"P5\n{width} {height}\n255\n".encode() + pixels


class FakeTools:
    """Stand in for VHS, FFmpeg, and gifsicle, recording each command."""

    def __init__(self, brightest: bytes) -> None:
        """Answer the brightest-frame request with the PGM ``brightest``."""
        self.brightest = brightest
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, *command: str) -> bytes:
        """Record ``command`` and produce what the real tool would."""
        self.commands.append(command)
        if command[0] == "ffmpeg":
            Path(command[-1]).write_bytes(
                self.brightest if "lagfun=decay=1" in command else b"cropped"
            )
        return b""


def test_docstring_examples_hold() -> None:
    # Act
    results = doctest.testmod(render)

    # Assert
    assert results.attempted > 0
    assert results.failed == 0


def test_parse_pgm_reads_the_size_and_pixels() -> None:
    # Arrange
    pgm = _pgm(3, 2, {(1, 0)})

    # Act
    image = render.parse_pgm(pgm)

    # Assert
    assert (image.width, image.height) == (3, 2)
    assert image.pixels == bytes([0, 200, 0, 0, 0, 0])


def test_parse_pgm_keeps_pixels_that_look_like_whitespace() -> None:
    # Arrange
    pgm = b"P5\n2 1\n255\n\x20\x0a"

    # Act
    image = render.parse_pgm(pgm)

    # Assert
    assert image.pixels == b"\x20\x0a"


@pytest.mark.parametrize("pgm", [b"P6\n1 1\n255\n\0", b"P5\n1 1\n65535\n\0\0", b""])
def test_parse_pgm_rejects_other_images(pgm: bytes) -> None:
    with pytest.raises(ValueError, match="PGM"):
        render.parse_pgm(pgm)


def test_content_box_bounds_every_lit_pixel() -> None:
    # Arrange
    image = render.parse_pgm(_pgm(10, 6, {(2, 1), (7, 3), (4, 4)}))

    # Act
    box = render.content_box(image)

    # Assert
    assert box == (2, 1, 8, 5)


def test_content_box_ignores_pixels_as_dark_as_the_background() -> None:
    # Arrange
    pixels = bytes([render.BACKGROUND_LIMIT, render.BACKGROUND_LIMIT + 1])
    image = render.Image(2, 1, pixels)

    # Act
    box = render.content_box(image)

    # Assert
    assert box == (1, 0, 2, 1)


def test_content_box_rejects_a_blank_video() -> None:
    with pytest.raises(ValueError, match="never draws"):
        render.content_box(render.Image(4, 4, bytes(16)))


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((32, 40, 1045, 300), (1078, 340)),
        ((32, 32, 1046, 300), (1078, 332)),
        ((32, 32, 1100, 360), (1120, 364)),
    ],
)
def test_crop_size_mirrors_the_margins_in_even_pixels_within_the_frame(
    box: tuple[int, int, int, int], expected: tuple[int, int]
) -> None:
    # Act
    size = render.crop_size(box, 1120, 364)

    # Assert
    assert size == expected


def test_render_records_the_tape_then_crops_both_videos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    mp4, gif = tmp_path / "demo.mp4", tmp_path / "demo.gif"
    tape = tmp_path / "demo.tape"
    tape.write_text(f"Output {mp4}\nOutput {gif}\nOutput {tmp_path / 'frames.txt'}\n")
    mp4.write_bytes(b"original")
    tools = FakeTools(_pgm(1120, 364, {(32, 40), (1045, 300)}))
    monkeypatch.setattr(render, "_run", tools)

    # Act
    size = render.render(tape)

    # Assert
    assert size == (1078, 342)
    assert tools.commands[0] == ("vhs", str(tape))
    crop = tools.commands[2]
    assert crop[crop.index("-vf") + 1] == "crop=1078:342:0:0"
    assert crop[-1] == str(mp4.with_suffix(".cropped.mp4"))
    assert mp4.read_bytes() == b"cropped"
    assert tools.commands[-1] == (
        "gifsicle", "--batch", "-O3", "--crop", "0,0+1078x342", str(gif),
    )  # fmt: skip


def test_render_rejects_a_tape_without_videos(tmp_path: Path) -> None:
    # Arrange
    tape = tmp_path / "frames.tape"
    tape.write_text("Output frames.txt\n")

    # Act / Assert
    with pytest.raises(ValueError, match="no MP4 or GIF"):
        render.render(tape)


def test_main_logs_a_failed_tool_and_returns_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    tape = tmp_path / "demo.tape"
    tape.write_text(f"Output {tmp_path / 'demo.gif'}\n")

    def fail(*command: str) -> bytes:
        raise subprocess.CalledProcessError(1, command, b"", b"no ttyd found")

    monkeypatch.setattr(render, "_run", fail)

    # Act
    with caplog.at_level(logging.ERROR):
        code = render.main([str(tape)])

    # Assert
    assert code == 1
    assert "vhs" in caplog.text
    assert "no ttyd found" in caplog.text
