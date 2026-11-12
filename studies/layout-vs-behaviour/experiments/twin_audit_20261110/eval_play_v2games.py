"""eval_play.py with the games of train/games_v2.py available. Same arguments."""
import sys
from pathlib import Path

OBS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OBS))
sys.path.insert(0, str(OBS / "train"))
import games_v2  # noqa: E402,F401
import eval_play  # noqa: E402

eval_play.main()
