"""
Options 2 + 3: Grammar-in-prompt + Validate-and-Retry
-------------------------------------------------------
Strategy:
  3. The system prompt includes a condensed version of the Triton grammar
     so the model self-steers during generation.
  2. After each generation we validate with Lark.  On failure we feed the
     parse error back and retry.

Dependencies:
    pip install lark openai google-generativeai
"""

from __future__ import annotations
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Callable

# ── Lark grammar (translated from the EBNF in the document) ───────────────────
# Lark uses EBNF with slight syntax differences.
# We encode the key structural rules; expression internals are approximated
# with a catch-all so the grammar stays tractable.

LARK_GRAMMAR = r"""
start: imports _NL* func_def (_NL* func_def)* _NL*

imports: "import triton" _NL "import triton.language as tl" _NL extra_import*
extra_import: ("import" | "from") /[^\n]+/ _NL

# ── Decorators ────────────────────────────────────────────────────────────────
func_def: decorator "def" WS NAME WS "(" WS paramlist WS ")" WS ":" WS _NL body_lines

decorator: autotune_dec heuristics_dec jit_dec
         | autotune_dec jit_dec
         | heuristics_dec jit_dec
         | jit_dec

autotune_dec:   "@triton.autotune(" /[^)]+/ ")" _NL
heuristics_dec: "@triton.heuristics(" /[^)]+/ ")" _NL
jit_dec:        "@triton.jit" ("(" /[^)]*/ ")")? _NL

# ── Parameters ────────────────────────────────────────────────────────────────
paramlist: (param ("," WS? param)* ","?)?
param: NAME WS? (":" WS? annotation)? (WS? "=" WS? default_val)?
     | "*" NAME | "**" NAME
annotation: NAME ("." NAME)*
default_val: /[^,)\n]+/

# ── Body: one or more indented lines (we validate indentation structurally) ───
body_lines: body_line+
body_line:  INDENT /[^\n]*/ _NL
          | _NL

INDENT: /[ \t]+/
WS: /[ \t]+/
NAME: /[a-zA-Z_][a-zA-Z0-9_]*/
_NL: /\r?\n/

%ignore /[ \t]+(?=\n)/   // trailing whitespace
"""

# System prompt fragment describing the grammar (Option 3)
GRAMMAR_SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert GPU kernel engineer writing Python code using the Triton framework.

Generate ONLY a complete, runnable Python source file.  The file MUST conform
to the following grammar (EBNF notation):

```
root       ::= imports func-def+
imports    ::= "import triton" NL
               "import triton.language as tl" NL
               extra-import*

decorator  ::= (autotune? heuristics? jit)
autotune   ::= "@triton.autotune(" args ")" NL
heuristics ::= "@triton.heuristics(" args ")" NL
jit        ::= "@triton.jit" ("(" args ")")? NL

func-def   ::= decorator "def" name "(" paramlist "):" NL body

param      ::= name (":" annotation)? ("=" expr)?
             | "*" name | "**" name
annotation ::= name ("." name)*

body       ::= (INDENT stmt NL)+   # 4-space or tab indented
```

Key rules:
- ALWAYS start with "import triton" then "import triton.language as tl".
- ALWAYS use @triton.jit (optionally combined with @triton.autotune / @triton.heuristics ABOVE it).
- Kernel body MUST be indented with exactly 4 spaces per level.
- Use tl.program_id, tl.load, tl.store, tl.constexpr etc. correctly.
- Do NOT output any prose, markdown fences, or explanation — only raw Python.
""")


# ── Validation ────────────────────────────────────────────────────────────────

def _get_parser():
    """Build and cache the Lark parser."""
    try:
        from lark import Lark
    except ImportError:
        raise ImportError("pip install lark")
    return Lark(LARK_GRAMMAR, parser="earley", ambiguity="resolve")


_PARSER = None


def validate(source: str) -> tuple[bool, str]:
    """
    Validate kernel source against the Lark grammar.
    Returns (ok, error_message).
    """
    global _PARSER
    if _PARSER is None:
        _PARSER = _get_parser()

    # Light pre-checks (fast, before full parse)
    if "import triton" not in source:
        return False, "Missing 'import triton'"
    if "@triton.jit" not in source:
        return False, "Missing @triton.jit decorator"
    if "def " not in source:
        return False, "Missing function definition"

    try:
        _PARSER.parse(source if source.endswith("\n") else source + "\n")
        return True, ""
    except Exception as exc:
        return False, str(exc)


# ── Extraction ────────────────────────────────────────────────────────────────

_CODE_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull the first fenced code block, or return the whole text."""
    m = _CODE_FENCE.search(text)
    return m.group(1).strip() if m else text.strip()


# ── Retry loop ────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    source: str
    attempts: int
    success: bool
    last_error: str = ""
    history: list[dict] = field(default_factory=list)


def _retry_loop(
    call_api: Callable[[list[dict]], str],
    prompt: str,
    max_attempts: int = 5,
) -> GenerationResult:
    """
    Core retry logic, backend-agnostic.
    `call_api` receives the full message history and returns the assistant text.
    """
    messages: list[dict] = [
        {"role": "system", "content": GRAMMAR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(1, max_attempts + 1):
        raw = call_api(messages)
        source = extract_code(raw)
        ok, err = validate(source)

        messages.append({"role": "assistant", "content": raw})

        if ok:
            return GenerationResult(
                source=source,
                attempts=attempt,
                success=True,
                history=messages,
            )

        # Feed the parse error back (Option 2 repair)
        repair_msg = (
            f"The code you produced failed grammar validation on attempt {attempt}:\n\n"
            f"  Error: {err}\n\n"
            "Please output a corrected version that strictly follows the grammar. "
            "Output ONLY raw Python, no markdown fences."
        )
        messages.append({"role": "user", "content": repair_msg})
        print(f"  [attempt {attempt}] parse error — retrying. {err[:120]}")

    return GenerationResult(
        source=source,
        attempts=max_attempts,
        success=False,
        last_error=err,
        history=messages,
    )


# ── OpenAI backend ────────────────────────────────────────────────────────────

def generate_openai(
    prompt: str,
    model: str = "gpt-4o",
    max_attempts: int = 5,
    api_key: str | None = None,
) -> GenerationResult:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    def call(messages: list[dict]) -> str:
        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content

    return _retry_loop(call, prompt, max_attempts)


# ── Gemini backend ────────────────────────────────────────────────────────────

def generate_gemini(
    prompt: str,
    model: str = "gemini-1.5-pro",
    max_attempts: int = 5,
    api_key: str | None = None,
) -> GenerationResult:
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("pip install google-generativeai")

    genai.configure(api_key=api_key or os.environ["GEMINI_API_KEY"])

    # Gemini doesn't natively accept a messages array, so we flatten to turns.
    def call(messages: list[dict]) -> str:
        # Separate system from conversation
        sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        history = [m for m in messages if m["role"] != "system"]

        gemini_model = genai.GenerativeModel(
            model_name=model, system_instruction=sys_msg
        )
        chat = gemini_model.start_chat(
            history=[
                {"role": m["role"], "parts": [m["content"]]}
                for m in history[:-1]  # all but the last
            ]
        )
        response = chat.send_message(history[-1]["content"])
        return response.text

    return _retry_loop(call, prompt, max_attempts)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a Triton kernel with grammar-prompted + validated retry."
    )
    parser.add_argument("--backend", choices=["openai", "gemini"], default="openai")
    parser.add_argument("--model", default=None, help="Override default model name")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument(
        "--prompt",
        default="Write a Triton kernel that adds two float32 vectors element-wise.",
    )
    args = parser.parse_args()

    kwargs = dict(prompt=args.prompt, max_attempts=args.attempts)
    if args.model:
        kwargs["model"] = args.model

    print(f"Backend: {args.backend}  |  max_attempts: {args.attempts}")
    print(f"Prompt: {args.prompt}\n")

    if args.backend == "openai":
        result = generate_openai(**kwargs)
    else:
        result = generate_gemini(**kwargs)

    print(f"\n{'✓ SUCCESS' if result.success else '✗ FAILED'} after {result.attempts} attempt(s)")
    if not result.success:
        print(f"Last error: {result.last_error}")

    print("\n=== Generated Kernel ===")
    print(result.source)
