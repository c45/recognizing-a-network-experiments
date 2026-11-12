"""train_game.py with the games of games_v2.py available. Same arguments."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import games_v2  # noqa: E402,F401  (adds the v2 games to games.GAMES)
import train_game  # noqa: E402

train_game.main()
