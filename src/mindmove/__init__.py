"""MindMove: imagined left/right hand decoding from prerecorded EEG."""

from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.version_info < (3, 12):
    raise RuntimeError("MindMove requires Python 3.12 or newer.")

_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

__version__ = "0.1.0"
