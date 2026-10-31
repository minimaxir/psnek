"""Record VHS tapes and crop their videos to the text they draw.

psnek sizes its bars to the terminal, so a tape is recorded wider than its text
needs, then cropped until the right and bottom margins mirror the left and top
ones. FFmpeg re-encodes the cropped MP4, and gifsicle crops the GIF losslessly.

Run from the repository root:

    python demo/render.py demo/demo.tape demo/tall.tape demo/hello.tape
"""

import logging
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

BACKGROUND_LIMIT = 40
_LIT = bytes(value > BACKGROUND_LIMIT for value in range(256))
CRF = 18
_OUTPUT = re.compile(r"^Output\s+(\S+)", re.MULTILINE)
_PGM_HEADER = re.compile(rb"P5\s+(\d+)\s+(\d+)\s+255\s")


class Image(NamedTuple):
    """An 8-bit grayscale image, stored row by row."""

    width: int
    height: int
    pixels: bytes


def parse_pgm(pgm: bytes) -> Image:
    """Parse a binary 8-bit PGM image.

    Args:
        pgm: A ``P5`` image with a maximum value of 255.

    Returns:
        Its size and pixels.

    Raises:
        ValueError: If ``pgm`` is not an 8-bit ``P5`` image.
    """
    header = _PGM_HEADER.match(pgm)
    width, height = map(int, header.groups()) if header else (0, 0)
    if not header or len(pgm) - header.end() != width * height:
        raise ValueError("expected an 8-bit binary PGM image")
    return Image(width, height, pgm[header.end() :])


def tape_outputs(tape: str) -> list[Path]:
    r"""Return the files a VHS tape writes, from its ``Output`` commands, in order.

    Example:
        >>> tape = 'Output demo/a.mp4\nOutput "demo/a.gif"\nType "ls"'
        >>> [path.as_posix() for path in tape_outputs(tape)]
        ['demo/a.mp4', 'demo/a.gif']
    """
    return [Path(path.strip("\"'")) for path in _OUTPUT.findall(tape)]


def content_box(image: Image) -> tuple[int, int, int, int]:
    """Bound the pixels brighter than the background.

    Args:
        image: The image to search.

    Returns:
        ``(left, top, right, bottom)``, exclusive on the right and bottom.

    Raises:
        ValueError: If every pixel is background.
    """
    lit = image.pixels.translate(_LIT)
    width = image.width
    spans = [
        (row, line.find(1), line.rfind(1) + 1)
        for row in range(image.height)
        if 1 in (line := lit[row * width : (row + 1) * width])
    ]
    if not spans:
        raise ValueError("the video never draws anything")
    return (
        min(left for _, left, _ in spans),
        spans[0][0],
        max(right for _, _, right in spans),
        spans[-1][0] + 1,
    )


def crop_size(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int]:
    """Size a top-left crop whose right and bottom margins mirror the left and top.

    Args:
        box: ``(left, top, right, bottom)`` of the content.
        width: Frame width.
        height: Frame height.

    Returns:
        Even ``(width, height)``, as 4:2:0 video requires, within the frame.

    Example:
        >>> crop_size((32, 40, 1045, 300), 1120, 364)
        (1078, 340)
    """
    left, top, right, bottom = box
    return (
        min(width, -(-(right + left) // 2) * 2),
        min(height, -(-(bottom + top) // 2) * 2),
    )


def _run(*command: str) -> bytes:
    return subprocess.run(command, check=True, capture_output=True).stdout  # noqa: S603


def brightest_frame(video: Path) -> Image:
    """Return each pixel's brightest value over every frame of ``video``."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder, "brightest.pgm")
        _run(
            "ffmpeg", "-v", "error", "-y", "-i", str(video),
            "-vf", "lagfun=decay=1", "-pix_fmt", "gray", "-update", "1", str(path),
        )  # fmt: skip
        return parse_pgm(path.read_bytes())


def crop_mp4(video: Path, width: int, height: int) -> None:
    """Crop an MP4 in place to its top-left ``width`` by ``height``.

    The frames are re-encoded, since browsers and macOS ignore the cropping
    fields in H.264 headers, at a slightly higher quality than VHS's CRF 20.
    """
    cropped = video.with_suffix(".cropped.mp4")
    _run(
        "ffmpeg", "-v", "error", "-y", "-i", str(video),
        "-vf", f"crop={width}:{height}:0:0", "-c:v", "libx264", "-crf", str(CRF),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(cropped),
    )  # fmt: skip
    cropped.replace(video)


def crop_gif(gif: Path, width: int, height: int) -> None:
    """Crop a GIF in place to its top-left ``width`` by ``height``, optimized."""
    _run("gifsicle", "--batch", "-O3", "--crop", f"0,0+{width}x{height}", str(gif))


CROPPERS: dict[str, Callable[[Path, int, int], None]] = {
    ".mp4": crop_mp4,
    ".gif": crop_gif,
}


def render(tape: Path) -> tuple[int, int]:
    """Record ``tape`` with VHS, then crop each of its videos to the drawn text.

    Args:
        tape: Path of the VHS tape, run from the current directory.

    Returns:
        The cropped ``(width, height)``.

    Raises:
        ValueError: If the tape writes no MP4 or GIF, or draws nothing.
        subprocess.CalledProcessError: If VHS, FFmpeg, or gifsicle fails.
    """
    videos = [
        path for path in tape_outputs(tape.read_text()) if path.suffix in CROPPERS
    ]
    if not videos:
        raise ValueError(f"{tape} writes no MP4 or GIF")
    _run("vhs", str(tape))
    image = brightest_frame(videos[0])
    size = crop_size(content_box(image), image.width, image.height)
    for video in videos:
        CROPPERS[video.suffix](video, *size)
    return size


def main(tapes: list[str]) -> int:
    """Render each tape, logging any failure.

    Args:
        tapes: Paths of the VHS tapes.

    Returns:
        The process exit code: 1 if any tape failed, else 0.
    """
    logging.basicConfig(format="%(message)s", level=logging.INFO)
    failed = False
    for tape in map(Path, tapes):
        try:
            width, height = render(tape)
        except subprocess.CalledProcessError as error:
            stderr = error.stderr.decode(errors="replace").strip()
            logger.error(f"{tape}: {' '.join(error.cmd)} failed: {stderr}")
            failed = True
        except (OSError, ValueError) as error:
            logger.error(f"{tape}: {error}")
            failed = True
        else:
            logger.info(f"{tape}: cropped to {width}x{height}")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
