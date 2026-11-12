"""Locations of inputs outside this folder; override with environment variables.

WR_UPSTREAM        the post's released logs (default ../external/lm_understand-logs, fetched by
                   ../external/fetch_upstream.sh)
WR_SANDBOX_PYTHON  Python used inside the analyst sandbox (default ../.venv-sandbox/bin/python)
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
LVB = HERE.parent
UPSTREAM = Path(os.environ.get("WR_UPSTREAM", LVB / "external" / "lm_understand-logs"))
EXP1 = UPSTREAM / "exp1_which_game"
PROMPT_TEMPLATE = EXP1 / "runs" / "pong" / "prompt.txt"
DESCRIPTION = EXP1 / "networks" / "description.txt"
SANDBOX_PY = Path(os.environ.get("WR_SANDBOX_PYTHON", LVB / ".venv-sandbox" / "bin" / "python"))
