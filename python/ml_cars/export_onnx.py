"""Export a saved MLP to ONNX, with normalization embedded and parity verification.

python -m ml_cars.export_onnx --input models/baseline.json --output models/baseline.onnx
Set ORT_DYLIB_PATH first (see ml_cars.onnx_runtime). No retraining is needed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from . import policy_actions

METADATA = {
    "ml_cars.version": "1",
    "ml_cars.observation_schema": "ml_cars/obs-v1",
    "ml_cars.action_schema": "throttle-steering/tanh-v1",
    "ml_cars.normalization": "embedded/v1",
    "ml_cars.recurrent": "false",
}


class NormalizedPolicy(nn.Module):
    def __init__(self, model, mean, scale):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))

    def forward(self, observations):
        return self.model((observations - self.mean) / self.scale)


def export_onnx(model, mean, scale, path):
    """Export a stateless float32 [1,18] -> [1,2] policy using opset 18."""
    import onnx
    path = Path(path)
    if path.suffix.lower() != ".onnx":
        raise ValueError("ONNX output must have the .onnx extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapped = NormalizedPolicy(model.cpu().eval(), mean, scale).eval()
    # A single file with embedded weights is straightforward to deploy with Rust.
    torch.onnx.export(wrapped, (torch.zeros(1, 18),), str(path),
                      input_names=["observations"], output_names=["actions"],
                      dynamo=True, opset_version=18, external_data=False, verify=True)
    graph = onnx.load(path)
    for key, value in METADATA.items():
        entry = graph.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(graph)
    onnx.save(graph, path)


def load_saved_model(path):
    # The existing Rust loader enforces versions, finite values and tensor shapes.
    policy_actions(str(path), np.empty((0, 18), dtype=np.float32))
    payload = json.loads(Path(path).read_text())
    layers = []
    for layer in payload["layers"]:
        weights = torch.tensor(layer["weights"], dtype=torch.float32)
        linear = nn.Linear(weights.shape[1], weights.shape[0])
        with torch.no_grad():
            linear.weight.copy_(weights)
            linear.bias.copy_(torch.tensor(layer["bias"], dtype=torch.float32))
        layers.extend([linear, nn.Tanh()])
    return nn.Sequential(*layers).eval(), np.array(payload["mean"], np.float32), np.array(payload["scale"], np.float32)


def main():
    from .train import collect, evaluate, verify
    import onnx
    import onnxruntime
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("models/baseline.json"))
    parser.add_argument("--output", type=Path, default=Path("models/baseline.onnx"))
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    torch.set_num_threads(1)
    model, mean, scale = load_saved_model(args.input)
    export_onnx(model, mean, scale, args.output)
    obs, _ = collect(4, 7)
    samples = obs[np.linspace(0, len(obs) - 1, 512, dtype=int)]
    # Includes the Rust binding and a standalone Rust process.
    error = verify(model, mean, scale, args.output, samples)
    evaluation = evaluate(args.output, 1007)
    report = {
        "source_policy": str(args.input),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "onnx_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "cargo_lock_sha256": hashlib.sha256(Path("Cargo.lock").read_bytes()).hexdigest(),
        "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__,
        "opset": 18, "input_shape": [1, 18], "output_shape": [1, 2], "normalization": "embedded",
        "parity_samples": len(samples), "parity_max_absolute_error": error, "parity_tolerance": 2e-5,
        "evaluation": evaluation,
    }
    args.output.with_suffix(".onnx.report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if any(r["reason"] != "success" for r in evaluation):
        raise RuntimeError("exported policy failed one or more route evaluations; see the report")


if __name__ == "__main__":
    main()
