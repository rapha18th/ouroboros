from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # .../ouroboros, which holds repo/ and data/
RAW = ROOT / "data" / "raw"
TRIPWIRE_RAW = RAW / "tripwire"
CVD_RAW = RAW / "cvd"
RENDERED = ROOT / "data" / "rendered"
