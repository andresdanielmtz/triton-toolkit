"""
compare_options.py
------------------
Run both Option 1 (JSON Schema) and Option 2+3 (grammar-prompted + retry)
against the same prompt and print a side-by-side report.

Usage:
    python compare_options.py --backend openai --prompt "..."
    python compare_options.py --backend gemini --prompt "..."
"""

import argparse
import time

from option1_json_schema import (
    generate_openai as o1_openai,
    generate_gemini as o1_gemini,
)
from option2_3_grammar_retry import (
    generate_openai as o23_openai,
    generate_gemini as o23_gemini,
)

SEP = "─" * 70


def run(backend: str, prompt: str, attempts: int) -> None:
    print(f"\n{'═' * 70}")
    print(f"  PROMPT : {prompt}")
    print(f"  BACKEND: {backend.upper()}")
    print(f"{'═' * 70}\n")

    # ── Option 1 ──────────────────────────────────────────────────────────────
    print("[ OPTION 1 — JSON Schema / Structured Output ]")
    print(SEP)
    t0 = time.perf_counter()
    try:
        if backend == "openai":
            raw, kernel = o1_openai(prompt)
        else:
            raw, kernel = o1_gemini(prompt)
        elapsed = time.perf_counter() - t0
        print(f"✓  Completed in {elapsed:.1f}s")
        print("\n--- Reconstructed kernel ---")
        print(kernel)
    except Exception as exc:
        print(f"✗  Failed: {exc}")
    print()

    # ── Option 2+3 ────────────────────────────────────────────────────────────
    print("[ OPTION 2+3 — Grammar Prompt + Validate & Retry ]")
    print(SEP)
    t0 = time.perf_counter()
    try:
        if backend == "openai":
            result = o23_openai(prompt, max_attempts=attempts)
        else:
            result = o23_gemini(prompt, max_attempts=attempts)
        elapsed = time.perf_counter() - t0
        status = "✓  SUCCESS" if result.success else "✗  FAILED"
        print(f"{status} — {result.attempts} attempt(s) in {elapsed:.1f}s")
        if not result.success:
            print(f"   Last error: {result.last_error}")
        print("\n--- Generated kernel ---")
        print(result.source)
    except Exception as exc:
        print(f"✗  Failed: {exc}")

    print(f"\n{'═' * 70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["openai", "gemini"], default="openai")
    parser.add_argument("--attempts", type=int, default=5,
                        help="Max retry attempts for Option 2+3")
    parser.add_argument(
        "--prompt",
        default="Write a Triton kernel that performs element-wise ReLU on a float32 vector.",
    )
    args = parser.parse_args()
    run(args.backend, args.prompt, args.attempts)
