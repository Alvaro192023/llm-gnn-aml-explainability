"""Pytest bootstrap: make the project root importable so ``import codigo...`` works
without installing the package (keeps CI free of the heavy ML dependencies)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
