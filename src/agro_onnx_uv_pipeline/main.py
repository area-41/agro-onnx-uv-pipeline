# src/agro_onnx_uv_pipeline/main.py
import argparse
import os
import sys
import time
import numpy as np
import onnxruntime as ort

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.onnx")


def ensure_model_exists(path: str) -> None:
    """Verifica se o modelo ONNX existe.

    Se ausente, tenta importar o PyTorch para gerá-lo.
    """
    if os.path.exists(path):
        return

    print(
        f"[Aviso] Modelo não encontrado em '{path}'. Tentando gerar via PyTorch..."
    )

    try:
        import torch
        import torch.nn as nn

        class LeafHealthClassifier(nn.Module):

            def __init__(self):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                    nn.Linear(32, 2),
                )

            def forward(self, x):
                return self.fc(x)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        model = LeafHealthClassifier()
        model.eval()

        dummy_input = torch.randn(1, 10)
        torch.onnx.export(
            model,
            dummy_input,
            path,
            export_params=True,
            opset_version=14,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
        )
        print(f"[Sucesso] Modelo exportado com sucesso para '{path}'.")

    except ImportError:
        print(
            f"[Erro Fatal] O arquivo '{path}' não existe e o PyTorch não está instalado no ambiente atual."
        )
        print(
            "Certifique-se de executar 'uv run --group dev python -m ...' para incluir as dependências de dev."
        )
        sys.exit(1)


def run_inference(model_path: str, batch_size: int = 5) -> np.ndarray:
    """Executa a inferência de alta performance utilizando apenas o ONNX Runtime."""
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        model_path, session_options, providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Simulação de lote de dados de sensores/imagens de campo (batch_size, 10 features)
    sample_data = np.random.randn(batch_size, 10).astype(np.float32)

    t0 = time.perf_counter()
    outputs = session.run([output_name], {input_name: sample_data})
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(
        f"[ONNX Runtime] Inferência concluída para lote de {batch_size} amostras em {elapsed_ms:.2f} ms"
    )
    return outputs[0]


def main() -> None:
    """Ponto de entrada do CLI registrado em [project.scripts]."""
    parser = argparse.ArgumentParser(
        description="Agro ONNX Pipeline - CLI de Inferência de Alta Performance"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Caminho para o arquivo .onnx",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Tamanho do lote de amostras para inferência",
    )

    args = parser.parse_args()

    print("=" * 60)
    print(" AGRO-ONNX-UV-PIPELINE :: SERVIÇO DE INFERÊNCIA (EDGE/PROD)")
    print("=" * 60)

    ensure_model_exists(args.model_path)
    predictions = run_inference(args.model_path, batch_size=args.batch_size)

    print("\nResultados das Predições (Logits):")
    print(predictions)
    print("=" * 60)


if __name__ == "__main__":
    main()