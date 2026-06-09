"""

Note: For the specific unit tests to run, the grammar must not allow imports.

- So use: func-def (newline* func-def)* newline*

- Instead of:
root ::= imports newline* func-def (newline* func-def)* newline*
imports ::= "import triton" newline "import triton.language as tl" newline extra-import*
extra-import ::= "import " name ("." name)* newline | "from " name ("." name)* " import " name newline 

"""

import xgrammar
from transformers import AutoTokenizer

# Load grammar and tokenizer
with open("triton_grammar.gbnf", "r") as f:
    grammar_str = f.read()

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base")
tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(tokenizer)
compiled = xgrammar.GrammarCompiler(tokenizer_info).compile_grammar(
    xgrammar.Grammar.from_ebnf(grammar_str)
)

# ─── Test cases ───────────────────────────────────────────────────────────────
# Format: (description, kernel_string, should_pass)

tests = [

    # ── Decorators ────────────────────────────────────────────────────────────
    ("simple @triton.jit", """@triton.jit
def kernel(x_ptr, n: tl.constexpr):
    x = tl.load(x_ptr)
""", True),

    ("@triton.jit with args", """@triton.jit(launch_metadata=fn)
def kernel(x_ptr, n: tl.constexpr):
    x = tl.load(x_ptr)
""", True),

    ("@triton.autotune + @triton.jit", """@triton.autotune(configs=[], key=['N'])
@triton.jit
def kernel(x_ptr, n: tl.constexpr):
    x = tl.load(x_ptr)
""", True),

    ("@triton.heuristics + @triton.jit", """@triton.heuristics(values={})
@triton.jit
def kernel(x_ptr, n: tl.constexpr):
    x = tl.load(x_ptr)
""", True),

    ("@triton.autotune + @triton.heuristics + @triton.jit", """@triton.autotune(configs=[], key=['N'])
@triton.heuristics(values={})
@triton.jit
def kernel(x_ptr, n: tl.constexpr):
    x = tl.load(x_ptr)
""", True),

    ("wrong decorator — should fail", """@torch.jit
def kernel(x_ptr):
    x = tl.load(x_ptr)
""", False),

    ("wrong order — should fail", """@triton.jit
@triton.autotune(configs=[], key=['N'])
def kernel(x_ptr):
    x = tl.load(x_ptr)
""", False),

    ("no decorator — should fail", """def kernel(x_ptr):
    x = tl.load(x_ptr)
""", False),

    # ── Params ────────────────────────────────────────────────────────────────
    ("tl.constexpr param", """@triton.jit
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    x = tl.load(x_ptr)
""", True),

    ("param with default", """@triton.jit
def kernel(x_ptr, n=128):
    x = tl.load(x_ptr)
""", True),

    ("annotated param with default", """@triton.jit
def kernel(x_ptr, n: tl.constexpr = 128):
    x = tl.load(x_ptr)
""", True),

    ("star param", """@triton.jit
def kernel(x_ptr, *args):
    x = tl.load(x_ptr)
""", True),

    # ── Assignments ───────────────────────────────────────────────────────────
    ("simple assignment", """@triton.jit
def kernel(x_ptr):
    x = tl.load(x_ptr)
""", True),

    ("augmented assignment +=", """@triton.jit
def kernel(x_ptr):
    x = tl.load(x_ptr)
    x += 1
""", True),

    ("tuple assignment", """@triton.jit
def kernel(x_ptr):
    a, b = tl.load(x_ptr), tl.load(x_ptr)
""", True),

    ("annotated assignment", """@triton.jit
def kernel(x_ptr):
    SCALE: tl.constexpr = 32
    x = tl.load(x_ptr)
""", True),

    ("subscript lvalue", """@triton.jit
def kernel(x_ptr):
    x = tl.load(x_ptr)
    x[0] = 1
""", True),

    ("attribute lvalue", """@triton.jit
def kernel(x_ptr):
    x = tl.load(x_ptr)
    x.dtype = tl.float32
""", True),

    # ── Control flow ──────────────────────────────────────────────────────────
    ("if statement", """@triton.jit
def kernel(x_ptr, n):
    if n > 0:
        x = tl.load(x_ptr)
""", True),

    ("if/elif/else", """@triton.jit
def kernel(x_ptr, n):
    if n > 0:
        x = tl.load(x_ptr)
    elif n == 0:
        x = tl.zeros([1], dtype=tl.float32)
    else:
        x = tl.load(x_ptr)
""", True),

    ("for loop", """@triton.jit
def kernel(x_ptr, n):
    for i in range(0, n):
        x = tl.load(x_ptr + i)
""", True),

    ("while loop", """@triton.jit
def kernel(x_ptr, n):
    while tl.atomic_cas(x_ptr, 0, 1) == 1:
        pass
""", True),

    ("with statement", """@triton.jit
def kernel(x_ptr):
    with tl.async_task(0):
        x = tl.load(x_ptr)
""", True),

    ("nested for + if (level 2)", """@triton.jit
def kernel(x_ptr, n, BLOCK_SIZE: tl.constexpr):
    for i in range(0, n):
        if i > 0:
            x = tl.load(x_ptr + i)
""", True),

    ("nested for + for + if (level 3)", """@triton.jit
def kernel(x_ptr, n, BLOCK_SIZE: tl.constexpr):
    for i in range(0, n):
        for j in range(0, n):
            if i > j:
                x = tl.load(x_ptr + i + j)
""", True),

    # ── Expressions ───────────────────────────────────────────────────────────
    ("ternary expression", """@triton.jit
def kernel(x_ptr, n):
    x = tl.load(x_ptr) if n > 0 else tl.zeros([1], dtype=tl.float32)
""", True),

    ("boolean operators", """@triton.jit
def kernel(x_ptr, n, m):
    if n > 0 and m > 0:
        x = tl.load(x_ptr)
""", True),

    ("bitwise operators", """@triton.jit
def kernel(x_ptr, n, m):
    mask = (n > 0) & (m > 0)
    x = tl.load(x_ptr, mask=mask)
""", True),

    ("not in operator", """@triton.jit
def kernel(x_ptr, n):
    if n not in [0, 1]:
        x = tl.load(x_ptr)
""", True),

    ("power operator", """@triton.jit
def kernel(x_ptr):
    x = tl.load(x_ptr) ** 2
""", True),

    ("lambda expression", """@triton.autotune(
    configs=[triton.Config({'BLOCK': b}) for b in [32, 64]],
    key=['N'],
)
@triton.jit
def kernel(x_ptr, N, BLOCK: tl.constexpr):
    x = tl.load(x_ptr)
""", True),

    # ── Collections ───────────────────────────────────────────────────────────
    ("list literal", """@triton.jit
def kernel(x_ptr):
    x = tl.zeros([32], dtype=tl.float32)
""", True),

    ("dict literal", """@triton.autotune(configs=[triton.Config({'BLOCK': 32})], key=['N'])
@triton.jit
def kernel(x_ptr, N, BLOCK: tl.constexpr):
    x = tl.load(x_ptr)
""", True),

    ("list comprehension", """@triton.autotune(
    configs=[triton.Config({'BLOCK': b}) for b in [32, 64, 128]],
    key=['N'],
)
@triton.jit
def kernel(x_ptr, N, BLOCK: tl.constexpr):
    x = tl.load(x_ptr)
""", True),

    # ── Indexing ──────────────────────────────────────────────────────────────
    ("simple subscript", """@triton.jit
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
""", True),

    ("2D subscript with None", """@triton.jit
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    x = offs[:, None] * offs[None, :]
""", True),

    ("slice", """@triton.jit
def kernel(x_ptr, BLOCK_SIZE: tl.constexpr):
    offs = tl.arange(0, BLOCK_SIZE)
    x = offs[1:10]
""", True),

    # ── Return ────────────────────────────────────────────────────────────────
    ("return single value", """@triton.jit
def kernel(x_ptr):
    x = tl.load(x_ptr)
    return x
""", True),

    ("return tuple", """@triton.jit
def kernel(x_ptr):
    x = tl.load(x_ptr)
    return x, x + 1
""", True),

    ("bare return", """@triton.jit
def kernel(x_ptr):
    x = tl.load(x_ptr)
    return
""", True),

    # ── Strings ───────────────────────────────────────────────────────────────
    ("f-string", """@triton.jit
def kernel(x_ptr, n):
    tl.device_print(f"n = {n}")
    x = tl.load(x_ptr)
""", True),

    ("long string docstring", """@triton.jit
def kernel(x_ptr, n):
    \"\"\"This is a docstring.\"\"\"
    x = tl.load(x_ptr)
""", True),

    # ── Multiple kernels ──────────────────────────────────────────────────────
    ("two kernels", """@triton.jit
def kernel1(x_ptr):
    x = tl.load(x_ptr)

@triton.jit
def kernel2(x_ptr):
    x = tl.load(x_ptr)
""", True),

    # ── Formatting ────────────────────────────────────────────────────────────
    ("empty line in block",
"@triton.jit\ndef kernel(x_ptr):\n    x = tl.load(x_ptr)\n\n    tl.store(x_ptr, x)\n",
True),

    ("trailing spaces",
"@triton.jit\ndef kernel(x_ptr):  \n    x = tl.load(x_ptr)  \n",
True),

    ("tab indentation",
"@triton.jit\ndef kernel(x_ptr):\n\tx = tl.load(x_ptr)\n",
True),

    # ── Invalid cases ─────────────────────────────────────────────────────────
    ("nested def — should fail", """@triton.jit
def kernel(x_ptr):
    def inner():
        pass
    x = tl.load(x_ptr)
""", False),

    ("assignment to call — should fail", """@triton.jit
def kernel(x_ptr):
    tl.load(x_ptr) = 1
""", False),

    ("import inside kernel — should fail", """@triton.jit
def kernel(x_ptr):
    import torch
    x = tl.load(x_ptr)
""", False),

]

# ─── Run tests ────────────────────────────────────────────────────────────────
passed = 0
failed = 0
errors = []

for desc, kernel, should_pass in tests:
    matcher = xgrammar.GrammarMatcher(compiled)
    # Ensure trailing newline without stripping content
    if not kernel.endswith("\n"):
        kernel = kernel + "\n"
    accepted = matcher.accept_string(kernel)

    if accepted == should_pass:
        print(f"  ✅ {desc}")
        passed += 1
    else:
        status = "accepted" if accepted else "rejected"
        print(f"  ❌ {desc} — was {status}, expected {'accepted' if should_pass else 'rejected'}")
        failed += 1
        errors.append(desc)

print()
print(f"Results: {passed}/{passed+failed} passed")
if errors:
    print(f"Failed: {errors}")