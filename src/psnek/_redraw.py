"""Escape sequences of the live region, and its rows as drawn cell by cell."""

import re
import unicodedata
from collections.abc import Iterable
from functools import lru_cache
from typing import NamedTuple

from psnek._style import PLAIN, Ink, Run
from psnek._text import EMOJI_PRESENTATION, char_width, display_width, disputed

SYNC_BEGIN, SYNC_END = "\x1b[?2026h", "\x1b[?2026l"
HIDE_CURSOR, SHOW_CURSOR = "\x1b[?25l", "\x1b[?25h"
WRAP_OFF, WRAP_ON = "\x1b[?7l", "\x1b[?7h"
COLUMN_ONE = "\x1b[1G"
PROGRESS_CLEAR = "\x1b]9;4;0;0\x1b\\"
# Skin tones, Hangul vowels and finals, kana and Thai sound marks, and Kirat
# Rai vowels join the character before them on some terminals; Hangul leading
# consonants and prepended marks such as Malayalam's dot reph the one after.
_JOINS_BEFORE = re.compile(
    r"[\U0001F3FB-\U0001F3FF\u1160-\u11ff\ud7b0-\ud7ff\uff9e\uff9f\u0e33\u0eb3"
    r"\U00016D63\U00016D67-\U00016D6A]"
)
_JOINS_AFTER = re.compile(
    r"[\u1100-\u115f\ua960-\ua97f\u0d4e\U000111C2\U000111C3\U0001193F\U00011941"
    r"\U000113D1\U00011A3A\U00011A84-\U00011A89\U00011D46\U00011F02]"
)

Placed = tuple[int, str, Ink]


class Drawn(NamedTuple):
    """A region row as drawn: each styled run at the column it starts in.

    ``patchable``: every run starts a cell; ``disputed``: runs of uncertain width.
    """

    placed: tuple[Placed, ...]
    width: int
    patchable: bool
    disputed: tuple[Placed, ...]

    @property
    def runs(self) -> list[Run]:
        """The row's runs, in order."""
        return [(text, ink) for _, text, ink in self.placed]


@lru_cache(maxsize=256)
def _own_cell(char: str) -> bool:
    """Whether a run starting with ``char`` starts a cell, not joining the last."""
    return (
        char != EMOJI_PRESENTATION
        and char_width(char) > 0
        and unicodedata.category(char) != "Mc"
        and _JOINS_BEFORE.match(char) is None
    )


def drawn(runs: Iterable[Run]) -> Drawn:
    """Place ``runs`` at their columns; a disputed run's leading spaces split off."""
    placed: list[Placed] = []
    unsure: list[Placed] = []
    column, patchable = 0, True
    for text, ink in runs:
        if not text:
            continue
        if not disputed(text):
            placed.append((column, text, ink if text.strip(" ") else PLAIN))
            column += display_width(text)
            continue
        body = text.lstrip(" ")
        if indent := len(text) - len(body):
            placed.append((column, text[:indent], PLAIN))
            column += indent
        run = (column, body, ink)
        placed.append(run)
        unsure.append(run)
        patchable &= _own_cell(body[0]) and _JOINS_AFTER.match(body[-1]) is None
        column += display_width(body)
    return Drawn(tuple(placed), column, patchable, tuple(unsure))
