import os
import torch

# from pytorch_impl import run_pytorch
# from triton_impl import run_triton


# Carpetas donde se guardan los resultados previamente generados
PYTORCH_FOLDER = "pytorch_results"
TRITON_FOLDER = "triton_results"


def compare_tensors(pytorch_output, triton_output, atol=1e-4, rtol=1e-4):
    is_correct = torch.allclose(
        pytorch_output,
        triton_output,
        atol=atol,
        rtol=rtol,
    )

    max_error = torch.max(
        torch.abs(pytorch_output - triton_output)
    ).item()

    return is_correct, max_error


# def compare_runtime_outputs():
#     """
#     Prueba individual.

#     Genera datos en el momento, ejecuta la implementación de PyTorch
#     y la implementación de Triton, y compara sus resultados.
#     """

#     if not torch.cuda.is_available():
#         raise RuntimeError("CUDA is not available. Triton requires an NVIDIA GPU.")

#     device = "cuda"

#     # Inputs de prueba
#     x = torch.randn(1024 * 1024, device=device)
#     y = torch.randn(1024 * 1024, device=device)

#     # Ejecuta implementación PyTorch
#     pytorch_output = run_pytorch(x, y)

#     # Ejecuta implementación Triton
#     triton_output = run_triton(x, y)

#     # Compara outputs
#     is_correct, max_error = compare_tensors(
#         pytorch_output,
#         triton_output,
#     )

#     print("\nRuntime Comparison")
#     print("-" * 50)
#     print("Outputs match:", is_correct)
#     print("Max error:", max_error)


def compare_saved_outputs():
    pytorch_files = sorted(os.listdir(PYTORCH_FOLDER))
    results = []

    for file_name in pytorch_files:
        if not file_name.endswith(".pt"):
            continue

        pytorch_path = os.path.join(PYTORCH_FOLDER, file_name)
        triton_path = os.path.join(TRITON_FOLDER, file_name)

        if not os.path.exists(triton_path):
            results.append({
                "file": file_name,
                "status": "Missing Triton output",
                "match": False,
                "max_error": None,
            })
            continue

        pytorch_output = torch.load(pytorch_path)
        triton_output = torch.load(triton_path)

        is_correct, max_error = compare_tensors(
            pytorch_output,
            triton_output,
        )

        results.append({
            "file": file_name,
            "status": "Compared",
            "match": is_correct,
            "max_error": max_error,
        })
    print("\nSaved Files Comparison")
    print("-" * 50)
 
    # resultados detallados por archivo
    for result in results: 
        print(f"File: {result['file']}")
        print(f"Status: {result['status']}")
        print(f"Match: {result['match']}")
        print(f"Max Error: {result['max_error']}")
        print("-" * 50)


def main():
    # compare_runtime_outputs() #individual
    compare_saved_outputs()     #grupal


if __name__ == "__main__":
    main()