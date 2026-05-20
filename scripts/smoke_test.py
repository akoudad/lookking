"""
Smoke test — run the full Lookking pipeline on sample queries WITHOUT starting
Telegram. Verifies agents + LLM + DL model wiring.

Run from project root:
    python3 scripts/smoke_test.py            # 1 query (fast verify)
    python3 scripts/smoke_test.py --full     # 3 queries with sleep between
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from agents.crew_setup import run_lookking


SAMPLES = [
    "[MODE: places] gym in Casablanca",
    "[MODE: leads] I offer video editing for restaurants in Rabat",
    "[MODE: places] luxury spa Marrakech",
]


def main():
    full = "--full" in sys.argv
    samples = SAMPLES if full else SAMPLES[:1]
    for i, query in enumerate(samples, 1):
        print("=" * 70)
        print(f"[{i}] QUERY: {query}")
        print("=" * 70)
        result = run_lookking(query)
        print(result)
        print()
        if full and i < len(samples):
            # Free-tier LLMs have per-minute caps; sleep to avoid hitting them.
            print("(sleeping 60s to respect per-minute LLM quota...)")
            time.sleep(60)


if __name__ == "__main__":
    main()
