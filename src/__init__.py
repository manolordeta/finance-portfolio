"""
Quantitative Investment System — src package.

Loads .env on import and ensures data directories exist.
"""

from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# Ensure data directories exist
(_PROJECT_ROOT / "data" / "cache").mkdir(parents=True, exist_ok=True)
(_PROJECT_ROOT / "data" / "db").mkdir(parents=True, exist_ok=True)
(_PROJECT_ROOT / "data" / "historical").mkdir(parents=True, exist_ok=True)
