"""
Test cases to simulate the TritonBench evaluation suite, but in a more controlled way to isolate specific syntax features and edge cases.
"""

import xgrammar
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME          = "deepseek-ai/deepseek-coder-1.3b-base"
MAX_NEW_TOKENS      = 512
DO_SAMPLE           = False
TEMPERATURE         = 0
REPETITION_PENALTY  = 1.3

# ─── Load model ───────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
print("✅ Model loaded!")

# ─── Load grammar ─────────────────────────────────────────────────────────────
print("Compiling grammar...")
with open("triton_grammar.gbnf", "r") as f:
    grammar_str = f.read()
tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(tokenizer)
compiled = xgrammar.GrammarCompiler(tokenizer_info).compile_grammar(
    xgrammar.Grammar.from_ebnf(grammar_str)
)
print("✅ Grammar compiled!")

# ─── Load system prompt ───────────────────────────────────────────────────────
with open("system_prompt.txt", "r") as f:
    system_prompt = f.read()

# ─── Test cases ───────────────────────────────────────────────────────────────
test_cases = [

    ("vector_add",
"""def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y
"""),

    ("softmax",
"""def softmax(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max(dim=-1, keepdim=True).values
    x = x - x_max
    x = torch.exp(x)
    x = x / x.sum(dim=-1, keepdim=True)
    return x
"""),

    ("layer_norm",
"""def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    return weight * x_hat + bias
"""),

    ("dropout",
"""def dropout(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    random = torch.rand_like(x)
    mask = random > p
    return torch.where(mask, x / (1 - p), torch.zeros_like(x))
"""),

    ("sum_reduction",
"""def sum_reduction(x: torch.Tensor) -> torch.Tensor:
    return torch.sum(x, dim=-1)
"""),

]

# ─── Generate ─────────────────────────────────────────────────────────────────
def generate_kernel(pytorch_code, use_cd=True):
    prompt = system_prompt + pytorch_code + "\n\nTriton:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
        temperature=TEMPERATURE,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
    )

    if use_cd:
        # Fresh processor for each generation — avoids state carry-over
        fresh_processor = xgrammar.contrib.hf.LogitsProcessor(compiled)
        kwargs["logits_processor"] = [fresh_processor]

    try:
        output = model.generate(**inputs, **kwargs)
        generated_ids = output[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)
    except AssertionError as e:
        return f"[CD_ERROR: Grammar matcher failed — {str(e)}]"
    except Exception as e:
        return f"[ERROR: {str(e)}]"

# ─── Check syntax ─────────────────────────────────────────────────────────────
def check_syntax(kernel_text):
    if kernel_text.startswith("["):
        return False
    if not kernel_text.endswith("\n"):
        kernel_text += "\n"
    matcher = xgrammar.GrammarMatcher(compiled)
    return matcher.accept_string(kernel_text)

# ─── Run evaluation ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)
print()
print("Generation config:")
print(f"  model:              {MODEL_NAME}")
print(f"  do_sample:          {DO_SAMPLE}")
print(f"  temperature:        {TEMPERATURE}")
print(f"  repetition_penalty: {REPETITION_PENALTY}")
print(f"  max_new_tokens:     {MAX_NEW_TOKENS}")
print(f"  grammar:            triton_grammar.gbnf")
print()

results = {
    "vanilla":    {"passed": 0, "failed": 0, "kernels": []},
    "vanilla_cd": {"passed": 0, "failed": 0, "kernels": []},
}

for op_name, pytorch_code in test_cases:
    print(f"{'─'*60}")
    print(f"Operator: {op_name}")
    print(f"PyTorch:\n{pytorch_code}")

    # ── Vanilla (no CD) ───────────────────────────────────────────────────────
    print("Generating vanilla...")
    vanilla_kernel = generate_kernel(pytorch_code, use_cd=False)
    vanilla_valid = check_syntax(vanilla_kernel)
    results["vanilla"]["kernels"].append((op_name, vanilla_kernel, vanilla_valid))
    if vanilla_valid:
        results["vanilla"]["passed"] += 1
        print(f"  Vanilla:    ✅ VALID")
    else:
        results["vanilla"]["failed"] += 1
        print(f"  Vanilla:    ❌ INVALID")
    print(f"  Generated:\n{vanilla_kernel[:300]}...")

    # ── Vanilla + CD ──────────────────────────────────────────────────────────
    print("Generating vanilla + CD...")
    cd_kernel = generate_kernel(pytorch_code, use_cd=True)
    cd_valid = check_syntax(cd_kernel)
    results["vanilla_cd"]["kernels"].append((op_name, cd_kernel, cd_valid))
    if cd_valid:
        results["vanilla_cd"]["passed"] += 1
        print(f"  Vanilla+CD: ✅ VALID")
    else:
        results["vanilla_cd"]["failed"] += 1
        print(f"  Vanilla+CD: ❌ INVALID")
    print(f"  Generated:\n{cd_kernel[:300]}...")
    print()

# ─── Summary ──────────────────────────────────────────────────────────────────
total        = len(test_cases)
vanilla_pass = results["vanilla"]["passed"]
cd_pass      = results["vanilla_cd"]["passed"]

print("="*60)
print("SUMMARY")
print("="*60)
print(f"  {'Condition':<20} {'Valid':>6} {'Invalid':>8} {'Score':>8}")
print(f"  {'─'*44}")
print(f"  {'Vanilla':<20} {vanilla_pass:>6} {total-vanilla_pass:>8} {vanilla_pass}/{total}")
print(f"  {'Vanilla + CD':<20} {cd_pass:>6} {total-cd_pass:>8} {cd_pass}/{total}")
print()

# ─── Save results ─────────────────────────────────────────────────────────────
with open("eval_results.txt", "w") as f:
    f.write("EVALUATION RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write("Generation config:\n")
    f.write(f"  model:              {MODEL_NAME}\n")
    f.write(f"  do_sample:          {DO_SAMPLE}\n")
    f.write(f"  repetition_penalty: {REPETITION_PENALTY}\n")
    f.write(f"  max_new_tokens:     {MAX_NEW_TOKENS}\n")
    f.write(f"  grammar:            triton_grammar.gbnf\n\n")
    f.write("="*60 + "\n\n")
    for condition in ["vanilla", "vanilla_cd"]:
        f.write(f"{condition.upper()}\n")
        f.write("-"*40 + "\n")
        for op_name, kernel, valid in results[condition]["kernels"]:
            f.write(f"Operator: {op_name}\n")
            f.write(f"Valid: {'YES' if valid else 'NO'}\n")
            f.write(f"Kernel:\n{kernel}\n\n")
        f.write("\n")
    f.write("SUMMARY\n")
    f.write("="*60 + "\n")
    f.write(f"  {'Condition':<20} {'Valid':>6} {'Invalid':>8} {'Score':>8}\n")
    f.write(f"  {'─'*44}\n")
    f.write(f"  {'Vanilla':<20} {vanilla_pass:>6} {total-vanilla_pass:>8} {vanilla_pass}/{total}\n")
    f.write(f"  {'Vanilla + CD':<20} {cd_pass:>6} {total-cd_pass:>8} {cd_pass}/{total}\n")

print("Results saved to eval_results.txt")