"""Test configuration for the rebuild package."""

import os
import sys
from pathlib import Path


SRC_DIR = Path(__file__).parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))
os.environ.setdefault("DEEPSEEK_API_KEY", "offline-test-key")
