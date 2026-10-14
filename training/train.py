#!/usr/bin/env python3
"""Train the fixed distortion baseline or condition plugin on a frozen Scaffold scene."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_scene import main

if __name__ == "__main__":
    raise SystemExit(main(training=True))
