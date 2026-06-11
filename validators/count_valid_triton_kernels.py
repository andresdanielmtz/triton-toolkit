"""
count_valid_triton_kernels.py

For each prediction JSONL file:
  1. Extracts / cleans the predicted code (strips markdown fences and prose).
  2. Checks whether the cleaned code contains @triton.jit.
  3. Reports valid kernel counts and a per-file summary table.

Usage:
    python count_valid_triton_kernels.py [file1.jsonl file2.jsonl ...]

If no files are given, the script scans the current working directory for
*.jsonl files and any files whose name starts with "predictions".
"""

import json
import re
import sys
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------

def extract_from_fences(text: str) -> str:
    """
    If the text contains markdown code fences (```python ... ``` or ``` ... ```),
    return the concatenated content of ALL complete fenced blocks.
    Returns an empty string if no complete blocks are found.
    """
    # Match ```python\n...\n``` or ```\n...\n```
    blocks = re.findall(r"```(?:python)?\n?(.*?)```", text, re.DOTALL)
    return "\n\n".join(blocks).strip()


def clean_prediction(raw: str) -> str:
    """
    Clean a raw model prediction down to just the Python/Triton code.

    Strategy:
      1. If the text contains complete markdown fences, extract their content
         (handles base-model outputs that wrap code in ```python ... ```).
      2. Otherwise use the raw text as-is (fine-tuned models output bare code).
    
    In both cases trailing/leading whitespace is stripped.
    """
    if "```" in raw:
        extracted = extract_from_fences(raw)
        if extracted:
            return extracted
        # Incomplete fence (generation was cut off) — strip the fence header
        # and return whatever code follows it.
        raw = re.sub(r"```(?:python)?", "", raw)

    return raw.strip()


# ---------------------------------------------------------------------------
# Validity check
# ---------------------------------------------------------------------------

VALID_PATTERN = re.compile(r"@triton\.jit")


def is_valid_kernel(code: str) -> bool:
    """Return True if the cleaned code contains @triton.jit."""
    return bool(VALID_PATTERN.search(code))


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(path: Path) -> dict:
    """
    Parse a JSONL file and return a results dict:
      {
        "file": str,
        "total": int,
        "valid": int,
        "invalid": int,
        "details": [ {"index": int, "valid": bool, "reason": str}, ... ]
      }
    """
    results = {
        "file": path.name,
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "details": [],
    }

    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                results["details"].append({
                    "index": idx,
                    "valid": False,
                    "reason": f"JSON parse error: {exc}",
                })
                results["total"] += 1
                results["invalid"] += 1
                continue

            raw = record.get("predict", "")
            cleaned = clean_prediction(raw)
            valid = is_valid_kernel(cleaned)

            results["total"] += 1
            if valid:
                results["valid"] += 1
                reason = "contains @triton.jit"
            else:
                results["invalid"] += 1
                if not cleaned:
                    reason = "empty prediction"
                else:
                    reason = "missing @triton.jit"

            results["details"].append({
                "index": idx,
                "valid": valid,
                "reason": reason,
            })

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(all_results: list[dict]) -> None:
    col_w = max(len(r["file"]) for r in all_results) + 2

    header = (
        f"{'File':<{col_w}}  {'Total':>6}  {'Valid':>6}  {'Invalid':>8}  {'% Valid':>8}"
    )
    sep = "-" * len(header)

    print("\n" + sep)
    print(header)
    print(sep)

    for r in all_results:
        pct = (r["valid"] / r["total"] * 100) if r["total"] else 0.0
        print(
            f"{r['file']:<{col_w}}  {r['total']:>6}  {r['valid']:>6}  "
            f"{r['invalid']:>8}  {pct:>7.1f}%"
        )

    print(sep)

    totals_total   = sum(r["total"]   for r in all_results)
    totals_valid   = sum(r["valid"]   for r in all_results)
    totals_invalid = sum(r["invalid"] for r in all_results)
    totals_pct     = (totals_valid / totals_total * 100) if totals_total else 0.0

    print(
        f"{'TOTAL':<{col_w}}  {totals_total:>6}  {totals_valid:>6}  "
        f"{totals_invalid:>8}  {totals_pct:>7.1f}%"
    )
    print(sep + "\n")


def print_invalid_details(results: dict) -> None:
    invalid = [d for d in results["details"] if not d["valid"]]
    if not invalid:
        return
    print(f"  Invalid predictions in {results['file']}:")
    for d in invalid:
        print(f"    row {d['index']:>4}: {d['reason']}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def discover_files(cwd: Path) -> list[Path]:
    """Find *.jsonl files or files matching predictions* in cwd."""
    found = sorted(
        set(cwd.glob("*.jsonl")) |
        set(cwd.glob("predictions*"))
    )
    return [f for f in found if f.is_file()]


def main() -> None:
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        cwd = Path(".")
        paths = discover_files(cwd)
        if not paths:
            print("No JSONL/prediction files found. Pass file paths as arguments.")
            sys.exit(1)
        print(f"Auto-discovered {len(paths)} file(s).")

    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: file not found: {p}", file=sys.stderr)
        sys.exit(1)

    for path in paths:
        res = process_file(path)

        print(
            f"{path.name}: "
            f"{res['valid']} valid / "
            f"{res['total']} total"
        )


if __name__ == "__main__":
    main()