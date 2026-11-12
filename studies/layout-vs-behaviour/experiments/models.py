"""Model keys of the reader arms and helpers shared by the scorers."""
from pathlib import Path

MODELS = ("astra", "opus")
CLAUDE = ("opus",)          # run through claude -p; run.json holds usage, effort.txt the effort used
DEFAULT_EFFORT = {"astra": "high", "opus": "max"}


def valid_answer(d, name="answer.md"):
    """Text of a finished reply, or None for a missing/empty reply, an archived attempt (<dir>.capped_<effort>_<time>),
    a reply that ended at the output cap, one marked UNANSWERED, or an error reply (usage limit, API error:
    run.json has is_error true and answer.md holds the error text)."""
    d = Path(d)
    f = d / name
    if "." in d.name or not f.exists():
        return None
    t = f.read_text()
    if not t.strip() or "output token maximum" in t or t.startswith("UNANSWERED"):
        return None
    run = d / "run.json"
    if run.exists() and '"is_error": true' in run.read_text():
        return None
    return t


def effort(d):
    d = Path(d)
    f = d / "effort.txt"
    return f.read_text().strip() if f.exists() else DEFAULT_EFFORT.get(d.name.split("_", 1)[0], "")
