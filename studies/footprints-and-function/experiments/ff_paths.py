"""Locations of inputs outside this folder; override with environment variables.

WR_LAYOUT_EXPERIMENTS  the layout study's experiments folder: trained Pong/Breakout/Pac-Man/Freeway policies,
                       camouflaged suites, shared prompt and scoring code (default ../../layout-vs-behaviour/experiments)
WR_UPSTREAM            the post's released logs (default ../external/lm_understand-logs, fetched by
                       ../external/fetch_upstream.sh)
WR_SANDBOX_PYTHON      Python used inside the analyst sandbox (default ../.venv-sandbox/bin/python)
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
FF = HERE.parent
LAYOUT = Path(os.environ.get("WR_LAYOUT_EXPERIMENTS", FF.parent / "layout-vs-behaviour" / "experiments"))
TRAIN = LAYOUT / "train"
UPSTREAM = Path(os.environ.get("WR_UPSTREAM", FF / "external" / "lm_understand-logs"))
EXP1 = UPSTREAM / "exp1_which_game"
EXP2 = UPSTREAM / "exp2_diffusion"
SANDBOX_PY = Path(os.environ.get("WR_SANDBOX_PYTHON", FF / ".venv-sandbox" / "bin" / "python"))
