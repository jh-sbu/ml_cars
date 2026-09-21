"""Reproducible behavior cloning baseline, portable export, and native parity check.

Run: python -m ml_cars.train --output models/baseline.json
Exports feedforward Linear/Tanh layers to JSON, with optional ONNX export.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tempfile
import numpy as np
import torch
from torch import nn
from . import Batch, OBS_DIM, policy_actions


def collect(episodes, seed):
    observations, actions = [], []
    for terrain in ("flat", "hills"):
        config = {"terrain": terrain, "spawn_jitter": 2., "vehicles": [{"id": "car_0", "controller": "native"}]}
        batch = Batch(json.dumps(config), list(range(seed, seed + episodes)))
        current = batch.observe()["observations"].copy()
        finished = np.zeros(episodes, dtype=bool)
        while not finished.all():
            result = batch.step_native()
            valid = ~finished
            observations.append(current[valid, 0].copy())
            actions.append(result["observations"][valid, 0, 6:8].copy())
            finished |= ~result["active"][:, 0]
            current = result["observations"].copy()
            # Keep batch slots live without mixing reset observations into final samples.
            for i in np.flatnonzero(~result["active"][:, 0]):
                current[i] = batch.reset_at(int(i), seed + int(i))["observations"][0]
    return np.concatenate(observations), np.concatenate(actions)


def export(model, mean, scale, path):
    layers = [{"weights": layer.weight.detach().tolist(), "bias": layer.bias.detach().tolist()} for layer in model if isinstance(layer, nn.Linear)]
    payload = {"version": 1, "observation_schema": "ml_cars/obs-v1", "action_schema": "throttle-steering/tanh-v1", "mean": mean.tolist(), "scale": scale.tolist(), "layers": layers}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def verify(model, mean, scale, path, observations):
    with torch.no_grad():
        expected = model(torch.from_numpy((observations - mean) / scale)).numpy()
    actual = policy_actions(str(path), np.ascontiguousarray(observations))
    max_error = float(np.max(np.abs(expected - actual)))
    if max_error > 2e-5:
        raise AssertionError(f"Rust/PyTorch parity error: {max_error}")
    # Also exercise a standalone Rust process, without a Python inference runtime.
    with tempfile.TemporaryDirectory() as directory:
        obs_path = Path(directory) / "observations.json"
        obs_path.write_text(json.dumps(observations[:32].tolist()))
        output = subprocess.check_output(["cargo", "run", "--quiet", "--", "infer", str(path), str(obs_path)], text=True)
        native = np.array([[a["throttle"], a["steering"]] for a in json.loads(output)], dtype=np.float32)
        np.testing.assert_allclose(native, expected[:32], atol=2e-5, rtol=0)
    return max_error


def evaluate(path, seed):
    results = []
    for terrain in ("flat", "hills"):
        config = {"terrain": terrain, "spawn_jitter": 2., "vehicles": [{"id": "car_0", "controller": "policy:" + str(path)}]}
        for episode_seed in range(seed, seed + 4):
            batch = Batch(json.dumps(config), [episode_seed])
            total = 0.
            while True:
                result = batch.step_native()
                total += float(result["rewards"][0, 0])
                if not result["active"][0, 0]:
                    break
            results.append({"terrain": terrain, "seed": episode_seed, "reason": result["reasons"][0][0], "reward": total})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/baseline.json"))
    parser.add_argument("--onnx-output", type=Path, help="also export a schema-checked ONNX policy (requires [onnx] dependencies)")
    parser.add_argument("--episodes", type=int, default=16, help="expert episodes per terrain")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.output.suffix.lower() != ".json":
        parser.error("--output must end in .json; use --onnx-output for ONNX")
    if args.onnx_output and args.onnx_output.suffix.lower() != ".onnx":
        parser.error("--onnx-output must end in .onnx")
    if args.episodes < 1 or args.epochs < 1:
        parser.error("episodes and epochs must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(args.seed)
    obs, target = collect(args.episodes, args.seed)
    mean, scale = obs.mean(axis=0), np.maximum(obs.std(axis=0), 0.1)
    x, y = torch.from_numpy((obs - mean) / scale), torch.from_numpy(target)
    model = nn.Sequential(nn.Linear(OBS_DIM, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2), nn.Tanh())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    for epoch in range(args.epochs):
        order = torch.randperm(len(x))
        for idx in order.split(512):
            prediction = model(x[idx])
            loss = ((prediction - y[idx]) ** 2 * torch.tensor([1., 4.])).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 40 == 0:
            print(f"epoch={epoch + 1} loss={loss.item():.6f}", flush=True)
    export(model, mean, scale, args.output)
    samples = obs[np.linspace(0, len(obs) - 1, 512, dtype=int)]
    max_error = verify(model, mean, scale, args.output, samples)
    evaluation = evaluate(args.output, args.seed + 1000)
    report = {"method": "behavior_cloning", "seed": args.seed, "samples": len(obs), "epochs": args.epochs, "expert_episodes_per_terrain": args.episodes, "torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform(), "cargo_lock_sha256": hashlib.sha256(Path("Cargo.lock").read_bytes()).hexdigest(), "policy_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "parity_samples": len(samples), "parity_max_absolute_error": max_error, "parity_tolerance": 2e-5, "evaluation": evaluation}
    if args.onnx_output:
        from .export_onnx import export_onnx
        export_onnx(model, mean, scale, args.onnx_output)
        report["onnx"] = {
            "path": str(args.onnx_output),
            "sha256": hashlib.sha256(args.onnx_output.read_bytes()).hexdigest(),
            "parity_max_absolute_error": verify(model, mean, scale, args.onnx_output, samples),
            "evaluation": evaluate(args.onnx_output, args.seed + 1000),
        }
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
