"""
run_pipeline.py

Run parser -> normalizer -> flag_engine end-to-end for one loan_id.
Each stage prints a header and is short-circuited on failure so you
can see which step broke.

    python run_pipeline.py --loan_id loan_001
    python run_pipeline.py --loan_id loan_002 --force
"""

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.parser import main as run_parser
from src.normalizer import main as run_normalizer
from src.flag_engine import main as run_flag_engine


def _step_banner(label: str) -> None:
    """Print a visible separator before each pipeline stage."""
    print()
    print("=" * 72)
    print(f"  {label}")
    print("=" * 72)


def main(loan_id: str, force: bool) -> int:
    """Run the three pipeline stages in order. Returns process exit code."""
    overall_start = time.perf_counter()

    print()
    print(f"Pipeline run for loan_id = {loan_id}")
    print(f"  force overwrite: {force}")

    # ---- 1. parser ----
    _step_banner("[1/3] PARSER — extract fields from PDFs with Groq")
    t0 = time.perf_counter()
    parsed = run_parser(loan_id=loan_id, force=force)
    if not parsed:
        print(f"\nABORTED: parser produced no results for {loan_id}.")
        print("If GROQ_API_KEY is missing, copy .env.example to .env and add your key.")
        return 1
    print(f"\n[1/3] complete — {len(parsed)} document(s) parsed in {time.perf_counter() - t0:.1f}s")

    # ---- 2. normalizer ----
    _step_banner("[2/3] NORMALIZER — merge per-document JSONs into a unified record")
    t0 = time.perf_counter()
    record = run_normalizer(loan_id=loan_id)
    if record is None:
        print(f"\nABORTED: normalizer failed for {loan_id}.")
        return 1
    print(f"\n[2/3] complete — record built in {time.perf_counter() - t0:.1f}s")

    # ---- 3. flag engine ----
    _step_banner("[3/3] FLAG ENGINE — evaluate underwriting rules")
    t0 = time.perf_counter()
    report = run_flag_engine(loan_id=loan_id)
    if report is None:
        print(f"\nABORTED: flag engine failed for {loan_id}.")
        return 1
    print(f"\n[3/3] complete — flags evaluated in {time.perf_counter() - t0:.1f}s")

    # ---- final summary ----
    elapsed = time.perf_counter() - overall_start
    print()
    print("=" * 72)
    print(f"  PIPELINE COMPLETE — {loan_id}")
    print("=" * 72)
    print(f"  Overall status:   {report['overall_status']}")
    print(f"  Flags raised:     {len(report['flags'])}")
    print(f"  Total elapsed:    {elapsed:.1f}s")
    print()
    return 0


def _cli() -> None:
    p = argparse.ArgumentParser(description="Run the full axia pipeline for one loan.")
    p.add_argument("--loan_id", default="loan_001")
    p.add_argument("--force", action="store_true",
                   help="Skip parser's re-run prompt and always overwrite existing parsed files.")
    args = p.parse_args()
    sys.exit(main(args.loan_id, args.force))


if __name__ == "__main__":
    _cli()
