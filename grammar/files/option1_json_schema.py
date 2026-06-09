"""
Option 1: JSON Schema / Structured Outputs
-------------------------------------------
Wraps the Triton kernel in a JSON envelope so OpenAI and Gemini
can enforce schema-level structure.  The schema captures what
JSON Schema *can* express from the grammar:
  - required top-level fields
  - known decorator combos as an enum
  - basic type constraints
What it cannot enforce: indentation levels, expression grammar,
specific argument shapes inside decorator calls.
"""

from __future__ import annotations
import json
import os
import textwrap
from typing import Any

# ── JSON Schema ────────────────────────────────────────────────────────────────

TRITON_KERNEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["imports", "kernel_name", "decorator", "parameters", "body"],
    "additionalProperties": False,
    "properties": {
        "imports": {
            "type": "array",
            "description": "Python import lines required by the kernel (excluding triton core imports which are always added).",
            "items": {"type": "string"},
        },
        "kernel_name": {
            "type": "string",
            "pattern": "^[a-zA-Z_][a-zA-Z0-9_]*$",
            "description": "Name of the @triton.jit function.",
        },
        "decorator": {
            "type": "object",
            "required": ["type"],
            "additionalProperties": False,
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [
                        "jit",
                        "autotune_jit",
                        "heuristics_jit",
                        "autotune_heuristics_jit",
                    ],
                    "description": "Which decorator combination to use.",
                },
                "autotune_configs": {
                    "type": "array",
                    "description": "List of triton.Config(...) dicts for @triton.autotune.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "kwargs": {"type": "object"},
                            "num_warps": {"type": "integer"},
                            "num_stages": {"type": "integer"},
                        },
                    },
                },
                "autotune_key": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Parameter names used as the autotune cache key.",
                },
                "heuristics": {
                    "type": "object",
                    "description": "Dict of param_name -> lambda-string for @triton.heuristics.",
                    "additionalProperties": {"type": "string"},
                },
                "jit_kwargs": {
                    "type": "object",
                    "description": "Extra kwargs passed to @triton.jit (e.g. interpret=True).",
                },
            },
        },
        "parameters": {
            "type": "array",
            "description": "Ordered list of kernel parameters.",
            "items": {
                "type": "object",
                "required": ["name"],
                "additionalProperties": False,
                "properties": {
                    "name": {"type": "string"},
                    "annotation": {
                        "type": "string",
                        "description": "Type annotation, e.g. 'tl.constexpr' or 'torch.Tensor'.",
                    },
                    "default": {
                        "type": "string",
                        "description": "Default value as a Python literal string.",
                    },
                },
            },
        },
        "body": {
            "type": "string",
            "description": (
                "Complete Python source of the kernel body, indented with 4 spaces. "
                "Must be valid Triton/Python. Do NOT include the def line or decorators."
            ),
        },
    },
}


# ── Reconstruction ─────────────────────────────────────────────────────────────

def _build_decorator(dec: dict[str, Any]) -> str:
    """Turn the decorator object back into Python source lines."""
    lines: list[str] = []
    dtype = dec["type"]

    if "autotune" in dtype:
        configs_src = ", ".join(
            "triton.Config({{{}}}, num_warps={}, num_stages={})".format(
                ", ".join(f'"{k}": {v}' for k, v in cfg.get("kwargs", {}).items()),
                cfg.get("num_warps", 4),
                cfg.get("num_stages", 2),
            )
            for cfg in dec.get("autotune_configs", [])
        )
        key_src = "[{}]".format(
            ", ".join(f'"{k}"' for k in dec.get("autotune_key", []))
        )
        lines.append(f"@triton.autotune(configs=[{configs_src}], key={key_src})")

    if "heuristics" in dtype:
        h_items = ", ".join(
            f'"{k}": {v}' for k, v in dec.get("heuristics", {}).items()
        )
        lines.append(f"@triton.heuristics({{{h_items}}})")

    jit_kwargs = dec.get("jit_kwargs") or {}
    if jit_kwargs:
        kw_src = ", ".join(f"{k}={v}" for k, v in jit_kwargs.items())
        lines.append(f"@triton.jit({kw_src})")
    else:
        lines.append("@triton.jit")

    return "\n".join(lines)


def reconstruct_kernel(data: dict[str, Any]) -> str:
    """Rebuild a complete, runnable Triton kernel from the JSON envelope."""
    parts: list[str] = [
        "import triton",
        "import triton.language as tl",
    ]
    for imp in data.get("imports") or []:
        parts.append(imp)

    parts.append("")
    parts.append(_build_decorator(data["decorator"]))

    params: list[str] = []
    for p in data["parameters"]:
        tok = p["name"]
        if ann := p.get("annotation"):
            tok += f": {ann}"
        if dflt := p.get("default"):
            tok += f" = {dflt}"
        params.append(tok)
    parts.append(f"def {data['kernel_name']}({', '.join(params)}):")

    body = textwrap.dedent(data["body"])
    for line in body.splitlines():
        parts.append("    " + line if line.strip() else line)

    return "\n".join(parts)


# ── OpenAI client ──────────────────────────────────────────────────────────────

def generate_openai(
    prompt: str,
    model: str = "gpt-4o",
    api_key: str | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Call OpenAI with structured output enforcement.
    Returns (raw_json_dict, reconstructed_kernel_source).
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert GPU kernel engineer. "
                    "Generate a valid Triton kernel that satisfies the user's request. "
                    "Return your answer strictly in the JSON format described by the schema. "
                    "The 'body' field must contain only the indented kernel body, "
                    "not the def line or decorators."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "triton_kernel",
                "strict": True,
                "schema": TRITON_KERNEL_SCHEMA,
            },
        },
    )

    raw = json.loads(response.choices[0].message.content)
    return raw, reconstruct_kernel(raw)


# ── Gemini client ──────────────────────────────────────────────────────────────

def generate_gemini(
    prompt: str,
    model: str = "gemini-1.5-pro",
    api_key: str | None = None,
) -> tuple[dict[str, Any], str]:
    """
    Call Gemini with structured output enforcement.
    Returns (raw_json_dict, reconstructed_kernel_source).
    """
    try:
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig
    except ImportError:
        raise ImportError("pip install google-generativeai")

    genai.configure(api_key=api_key or os.environ["GEMINI_API_KEY"])
    gemini_model = genai.GenerativeModel(
        model_name=model,
        system_instruction=(
            "You are an expert GPU kernel engineer. "
            "Generate a valid Triton kernel that satisfies the user's request. "
            "Return your answer strictly as JSON matching the provided schema. "
            "The 'body' field must contain only the indented kernel body."
        ),
    )

    response = gemini_model.generate_content(
        prompt,
        generation_config=GenerationConfig(
            response_mime_type="application/json",
            response_schema=TRITON_KERNEL_SCHEMA,
        ),
    )

    raw = json.loads(response.text)
    return raw, reconstruct_kernel(raw)


# ── Quick smoke test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["openai", "gemini"], default="openai")
    parser.add_argument(
        "--prompt",
        default="Write a Triton kernel that adds two float32 vectors element-wise.",
    )
    args = parser.parse_args()

    if args.backend == "openai":
        raw, kernel = generate_openai(args.prompt)
    else:
        raw, kernel = generate_gemini(args.prompt)

    print("=== Raw JSON ===")
    print(json.dumps(raw, indent=2))
    print("\n=== Reconstructed Kernel ===")
    print(kernel)
