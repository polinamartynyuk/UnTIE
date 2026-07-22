"""Совместимая точка входа для независимого от моделей пакета UnTIE."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from untie.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
