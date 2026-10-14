#!/usr/bin/env python3
"""Render a supplied or newly trained Scaffold condition plugin without training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_scene import main

if __name__ == "__main__":
    raise SystemExit(main(forced_action="inference"))
