# src/benchmark.py
import gc
import os
import statistics
import time
import tracemalloc
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


class LeafHealthClassifier(nn.Module):
    """Modelo de exemplo para classificação de saúde de folhas/pragas."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def measure_pytorch(
    model: nn.Module, inputs: torch.Tensor, runs: int = 1000
) -> tuple[list[float], float]:
    model.eval()

    # Warm-up para descartar overhead inicial de inicialização
    with torch.no_grad():
        for _ in range(50):
            _ = model(inputs)

    tracemalloc.start()
    gc.collect()

    times_ms = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(inputs)
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)

    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return times_ms, peak_mem_bytes / (1024 * 1024)


def measure_onnx(
    model_path: str, inputs_np: np.ndarray, runs: int = 1000
) -> tuple[list[float], float]:
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    session = ort.InferenceSession(
        model_path, session_options, providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_dict = {input_name: inputs_np}

    # Warm-up
    for _ in range(50):
        _ = session.run([output_name], input_dict)

    tracemalloc.start()
    gc.collect()

    times_ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = session.run([output_name], input_dict)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return times_ms, peak_mem_bytes / (1024 * 1024)


def calculate_stats(times_ms: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(times_ms),
        "std": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "p50": statistics.median(times_ms),
        "p95": float(np.percentile(times_ms, 95)),
        "p99": float(np.percentile(times_ms, 99)),
    }


def print_report(
    pt_stats: dict,
    pt_mem: float,
    onnx_stats: dict,
    onnx_mem: float,
    batch_size: int,
    runs: int,
):
    speedup = (
        pt_stats["mean"] / onnx_stats["mean"] if onnx_stats["mean"] > 0 else 0
    )
    mem_reduction = (
        ((pt_mem - onnx_mem) / pt_mem * 100) if pt_mem > 0 else 0.0
    )

    print("=" * 72)
    print(
        f" BENCHMARK: PyTorch vs ONNX Runtime (Batch Size: {batch_size}, Execuções: {runs})"
    )
    print("=" * 72)
    print(f"{'Métrica':<25} | {'PyTorch Nativo':<18} | {'ONNX Runtime':<18}")
    print("-" * 72)
    print(
        f"{'Latência Média (ms)':<25} | {pt_stats['mean']:<18.4f} | {onnx_stats['mean']:<18.4f}"
    )
    print(
        f"{'Desvio Padrão (ms)':<25} | {pt_stats['std']:<18.4f} | {onnx_stats['std']:<18.4f}"
    )
    print(
        f"{'Mediana / P50 (ms)':<25} | {pt_stats['p50']:<18.4f} | {onnx_stats['p50']:<18.4f}"
    )
    print(
        f"{'Percentil 95 / P95 (ms)':<25} | {pt_stats['p95']:<18.4f} | {onnx_stats['p95']:<18.4f}"
    )
    print(
        f"{'Percentil 99 / P99 (ms)':<25} | {pt_stats['p99']:<18.4f} | {onnx_stats['p99']:<18.4f}"
    )
    print(
        f"{'Pico Memória RAM (MB)':<25} | {pt_mem:<18.4f} | {onnx_mem:<18.4f}"
    )
    print("=" * 72)
    print(" RESUMO DOS GANHOS:")
    print(f"  • Speedup (Latência):  {speedup:.2f}x mais rápido no ONNX")
    print(
        f"  • Economia de Memória: {mem_reduction:.2f}% de redução no pico de RAM"
    )
    print("=" * 72)


def main():
    onnx_path = "models/model.onnx"

    # Exporta o modelo automaticamente caso o arquivo ainda não exista
    if not os.path.exists(onnx_path):
        os.makedirs("models", exist_ok=True)
        print(
            "Modelo ONNX não encontrado em 'models/model.onnx'. Gerando arquivo..."
        )
        model = LeafHealthClassifier()
        model.eval()
        dummy_input = torch.randn(1, 10)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )

    batch_size = 100
    runs = 1000
    num_features = 10

    data_np = np.random.randn(batch_size, num_features).astype(np.float32)
    data_tensor = torch.from_numpy(data_np)

    pt_model = LeafHealthClassifier()

    print(f"Iniciando benchmark ({runs} iterações, batch_size={batch_size})...")
    pt_times, pt_mem = measure_pytorch(pt_model, data_tensor, runs=runs)
    onnx_times, onnx_mem = measure_onnx(onnx_path, data_np, runs=runs)

    pt_stats = calculate_stats(pt_times)
    onnx_stats = calculate_stats(onnx_times)

    print_report(pt_stats, pt_mem, onnx_stats, onnx_mem, batch_size, runs)


if __name__ == "__main__":
    main()