"""Find the answer's sections (Game, Description, ...) whatever the heading style.

game_of(answer): the pre-registered keyword rule of obscured/score_suite.py, applied to the Game section. Headings may
be "Game: text", "**Game:** text", a "Game" heading line followed by text, or "**Game**" followed by text on the next
line. Falls back to the first 300 characters, as the original does.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ff_paths import LAYOUT  # noqa: E402
sys.path.insert(0, str(LAYOUT))
from score_suite import GAME_RULES  # noqa: E402

HEAD = r"(?mi)^[\W_]*(?:\d+[.)]\s*)?[\W_]*{name}[\W_]*?(?::[\W_]*(?P<a>\S.*)|[\W_]*$\n+(?P<b>.+))"


def game_line(answer):
    m = re.search(HEAD.format(name="Game"), answer)
    return (m.group("a") or m.group("b")) if m else None


def game_of(answer):
    line = (game_line(answer) or answer[:300]).lower()
    for g, pat in GAME_RULES:
        if re.search(pat, line):
            return g, line[:120]
    return "other", line[:120]
