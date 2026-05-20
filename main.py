"""
Lookking — Entry Point
Run: python3 main.py
"""
import sys
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "model" / "lookking_model.pt"
DATA_PATH  = Path(__file__).parent / "data" / "places.csv"


def check_setup():
    errors = []
    if not DATA_PATH.exists():
        errors.append("Data not generated. Run: python3 data/generate_data.py")
    if not MODEL_PATH.exists():
        errors.append("Model not trained. Run: python3 model/train.py")
    if errors:
        print("⚠️  Setup incomplete:")
        for e in errors:
            print(f"   → {e}")
        sys.exit(1)


if __name__ == "__main__":
    check_setup()
    from bot.telegram_bot import run_bot
    run_bot()
