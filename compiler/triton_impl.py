import torch
import triton
import triton.language as tl


BLOCK_SIZE = 1024


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    result = x + y
    tl.store(out_ptr + offsets, result, mask=mask)


def run_triton(x, y):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda _: (triton.cdiv(n_elements, BLOCK_SIZE),)
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
    )

    return output