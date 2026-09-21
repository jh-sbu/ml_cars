"""ONNX integration checks; install [onnx] and set ORT_DYLIB_PATH to run them."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("torch")
pytest.importorskip("onnxscript")
pytestmark = pytest.mark.skipif(not os.environ.get("ORT_DYLIB_PATH"), reason="set ORT_DYLIB_PATH for ONNX integration checks")
from ml_cars import Batch, policy_actions
from ml_cars.export_onnx import METADATA, load_saved_model, export_onnx

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/baseline.onnx"
JSON_MODEL = ROOT / "models/baseline.json"


def constant_policy(tmp_path, *, width=18, dtype=onnx.TensorProto.FLOAT,
                    output_width=2, batch_size=1, input_name="observations", extra_input=False,
                    metadata=None, values=None):
    inputs = [onnx.helper.make_tensor_value_info(input_name, dtype, [batch_size, width])]
    if extra_input:
        inputs.append(onnx.helper.make_tensor_value_info("hidden_state", onnx.TensorProto.FLOAT, [1, 4]))
    outputs = [onnx.helper.make_tensor_value_info("actions", onnx.TensorProto.FLOAT, [1, output_width])]
    tensor = onnx.helper.make_tensor("value", onnx.TensorProto.FLOAT, [1, output_width], values if values is not None else [0.] * output_width)
    node = onnx.helper.make_node("Constant", [], ["actions"], value=tensor)
    model = onnx.helper.make_model(onnx.helper.make_graph([node], "contract-test", inputs, outputs),
                                   opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=10)
    onnx.helper.set_model_props(model, METADATA if metadata is None else metadata)
    path = tmp_path / "constant.onnx"
    onnx.save(model, path)
    return str(path)


def test_saved_onnx_matches_json_and_pytorch():
    import torch
    from ml_cars.export_onnx import NormalizedPolicy
    model, mean, scale = load_saved_model(JSON_MODEL)
    rng = np.random.default_rng(42)
    obs = np.ascontiguousarray(mean + rng.normal(size=(128, 18)).astype(np.float32) * scale)
    with torch.no_grad():
        expected = NormalizedPolicy(model, mean, scale)(torch.from_numpy(obs)).numpy()
    actual = policy_actions(str(MODEL), obs)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)
    np.testing.assert_allclose(actual, policy_actions(str(JSON_MODEL), obs), atol=2e-5, rtol=0)


def test_export_round_trip(tmp_path):
    model, mean, scale = load_saved_model(JSON_MODEL)
    path = tmp_path / "exported.onnx"
    export_onnx(model, mean, scale, path)
    obs = np.zeros((4, 18), np.float32)
    np.testing.assert_allclose(policy_actions(str(path), obs), policy_actions(str(JSON_MODEL), obs), atol=2e-5, rtol=0)


@pytest.mark.parametrize("kwargs", [
    {"width": 17}, {"batch_size": 2}, {"dtype": onnx.TensorProto.DOUBLE}, {"output_width": 3},
    {"input_name": "wrong_name"}, {"extra_input": True}, {"metadata": {}},
    {"metadata": {**METADATA, "ml_cars.observation_schema": "wrong-order"}},
    {"metadata": {**METADATA, "ml_cars.normalization": "external"}},
    {"metadata": {**METADATA, "ml_cars.recurrent": "true"}},
])
def test_invalid_model_contract_rejected(tmp_path, kwargs):
    path = constant_policy(tmp_path, **kwargs)
    with pytest.raises(ValueError, match="ONNX"):
        policy_actions(path, np.zeros((1, 18), np.float32))


@pytest.mark.parametrize("values", [[2., 0.], [float("nan"), 0.], [0., float("inf")]])
def test_invalid_output_is_an_error_and_does_not_step_world(tmp_path, values):
    path = constant_policy(tmp_path, values=values)
    batch = Batch(json.dumps({"vehicles": [{"id": "bad", "controller": "policy:" + path}]}), [0])
    before = batch.snapshot_json(0)
    with pytest.raises(ValueError, match="invalid ONNX output"):
        batch.step_native()
    assert batch.snapshot_json(0) == before


def test_invalid_observation_and_corrupt_model(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        policy_actions(str(MODEL), np.full((1, 18), np.nan, np.float32))
    bad = tmp_path / "broken.onnx"
    bad.write_bytes(b"this is not an ONNX model")
    with pytest.raises(ValueError, match="ONNX"):
        policy_actions(str(bad), np.zeros((1, 18), np.float32))


def test_onnx_batch_matches_serial_and_resets():
    config = json.loads((ROOT / "scenarios/onnx-mixed.json").read_text())
    for v in config["vehicles"]:
        if v["controller"].startswith("policy:"):
            v["controller"] = "policy:" + str(ROOT / v["controller"].removeprefix("policy:"))
    config = json.dumps(config)
    batch = Batch(config, [4, 5])
    serial = [Batch(config, [seed]) for seed in [4, 5]]
    initial = batch.observe()["observations"].copy()
    for _ in range(40):
        result = batch.step_native()
        for i, single in enumerate(serial):
            expected = single.step_native()
            np.testing.assert_array_equal(result["observations"][i], expected["observations"][0])
            np.testing.assert_array_equal(result["rewards"][i], expected["rewards"][0])
    np.testing.assert_array_equal(batch.reset([4, 5])["observations"], initial)


def test_missing_runtime_is_actionable_and_json_still_works():
    # A fresh process is required: ONNX Runtime initialization is process-global.
    code = """
import numpy as np
from ml_cars import policy_actions
policy_actions(%r, np.zeros((1,18), np.float32))
try:
    policy_actions(%r, np.zeros((1,18), np.float32))
except ValueError as error:
    assert 'ORT_DYLIB_PATH' in str(error)
    assert 'ONNX Runtime unavailable' in str(error)
else:
    raise AssertionError('missing runtime was accepted')
""" % (str(JSON_MODEL), str(MODEL))
    env = {**os.environ, "ORT_DYLIB_PATH": "/nonexistent/ml-cars-runtime.so"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env, capture_output=True, text=True)


def test_dynamic_batch_dimension_accepts_single_vehicle(tmp_path):
    path = constant_policy(tmp_path, batch_size="batch")
    np.testing.assert_array_equal(policy_actions(path, np.zeros((3, 18), np.float32)), np.zeros((3, 2), np.float32))
