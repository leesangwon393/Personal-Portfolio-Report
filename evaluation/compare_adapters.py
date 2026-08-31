"""Section 23: Base vs LoRA comparison, retrieval held fixed.

Runs model_inference.py once per generation model (base Llama / FiQA LoRA /
TFNS LoRA / any other adapter under train_and_inference/) for the same
ticker + investor style + top_k. retrieve_news_with_fallback() is a pure
function of (ticker, investor_style, top_k, db state), so calling
model_inference.py multiple times with identical --ticker/--style/--top-k
against an unchanged db/news.db reproduces the exact same retrieval
context every time — only --adapter changes, isolating the comparison to
the generation model.

Each run's stdout (the report) and stderr (--debug retrieval trace) are
saved under evaluation/outputs/ for side-by-side reading.

Usage:
    python3 evaluation/compare_adapters.py --ticker NVDA --style SAFE
    python3 evaluation/compare_adapters.py --ticker NVDA --style AGGRESSIVE \\
        --adapters base fiqa tfns
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def run_one(ticker: str, style: str, adapter: str, top_k: int) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{ticker.upper()}_{style.upper()}_{adapter}.txt"
    cmd = [
        sys.executable, str(ROOT_DIR / "model_inference.py"),
        "--ticker", ticker, "--style", style, "--adapter", adapter,
        "--top-k", str(top_k), "--debug",
    ]
    result = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    out_path.write_text(
        "=== stdout (report) ===\n" + result.stdout +
        "\n=== stderr (retrieval debug) ===\n" + result.stderr,
        encoding="utf-8",
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--style", default="NEUTRAL")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--adapters", nargs="*", default=["base", "fiqa", "tfns"])
    args = parser.parse_args()

    for adapter in args.adapters:
        print(f"=== Running adapter={adapter} ===")
        out_path = run_one(args.ticker, args.style, adapter, args.top_k)
        print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
