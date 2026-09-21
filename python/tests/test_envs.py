import json
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test
from ml_cars import Batch, DrivingEnv, ParallelDrivingEnv


def test_gymnasium_contract():
    check_env(DrivingEnv(), skip_render_check=True)


def test_parallel_contract():
    config = {"max_steps": 30, "vehicles": [
        {"id": "learner", "controller": "external", "spawn": [0, 0]},
        {"id": "truck", "kind": "truck", "controller": "external", "spawn": [4, 10]},
        {"id": "traffic", "controller": "native", "spawn": [0, 20]}]}
    parallel_api_test(ParallelDrivingEnv(config), num_cycles=100)


def test_final_observation_and_reset_determinism():
    env = DrivingEnv({"max_steps": 1, "vehicles": [{"id": "car_0", "controller": "external"}]})
    start, _ = env.reset(seed=12)
    obs, _, terminated, truncated, info = env.step(np.zeros(2))
    assert not terminated and truncated
    np.testing.assert_array_equal(obs, info["final_observation"])
    with pytest.raises(RuntimeError):
        env.step(np.zeros(2))
    again, _ = env.reset(seed=12)
    np.testing.assert_array_equal(start, again)


def test_batch_ownership_shape_and_atomic_validation():
    config = json.dumps({"vehicles": [{"id": "car_0", "controller": "external"}]})
    batch = Batch(config, [2, 3])
    first = batch.observe()["observations"]
    original = first.copy()
    invalid = np.zeros((2, 1, 2), np.float32)
    invalid[1, 0, 0] = np.nan
    with pytest.raises(ValueError):
        batch.step(invalid)
    np.testing.assert_array_equal(original, batch.observe()["observations"])
    batch.step(np.ones((2, 1, 2), np.float32) * 0.2)
    np.testing.assert_array_equal(original, first)
    with pytest.raises(ValueError):
        batch.step(np.zeros((2, 2, 2), np.float32))


def test_parallel_removal_retains_physical_vehicle():
    config = {"max_steps": 2, "vehicles": [{"id": "a", "controller": "external"}, {"id": "b", "controller": "external", "spawn": [5, 10]}]}
    env = ParallelDrivingEnv(config)
    env.reset(seed=3)
    with pytest.raises(ValueError):
        env.step({"a": np.zeros(2)})
    for _ in range(2):
        result = env.step({a: np.zeros(2) for a in env.agents})
    assert not env.agents
    assert all(result[3].values())
    assert len(json.loads(env.batch.snapshot_json(0))["vehicles"]) == 2
    assert env.step({}) == ({}, {}, {}, {}, {})


def test_exported_policy_parity_on_normalized_samples():
    torch = pytest.importorskip("torch")
    from pathlib import Path
    from ml_cars import policy_actions
    path = Path(__file__).resolve().parents[2] / "models" / "baseline.json"
    model = json.loads(path.read_text())
    rng = np.random.default_rng(42)
    mean, scale = np.array(model["mean"], np.float32), np.array(model["scale"], np.float32)
    observations = np.ascontiguousarray(mean + rng.normal(size=(128, 18)).astype(np.float32) * scale)
    x = torch.from_numpy((observations - mean) / scale)
    for layer in model["layers"]:
        x = torch.tanh(x @ torch.tensor(layer["weights"]).T + torch.tensor(layer["bias"]))
    actual = policy_actions(str(path), observations)
    np.testing.assert_allclose(actual, x.numpy(), atol=2e-5, rtol=0)


def test_agent_can_finish_before_other_agents():
    config = {"max_steps": 50, "spawn_jitter": 0, "vehicles": [
        {"id": "near_finish", "controller": "external", "spawn": [0, 85]},
        {"id": "waiting", "controller": "external", "spawn": [0, 0]}]}
    env = ParallelDrivingEnv(config)
    env.reset(seed=0)
    for _ in range(49):
        result = env.step({a: np.array([1, 0] if a == "near_finish" else [0, 0], np.float32) for a in env.agents})
        if "near_finish" not in env.agents:
            break
    assert env.agents == ["waiting"]
    assert result[2]["near_finish"]
    assert "final_observation" in result[4]["near_finish"]
    result = env.step({"waiting": np.zeros(2, np.float32)})
    assert set(result[0]) == {"waiting"}
    assert len(json.loads(env.batch.snapshot_json(0))["vehicles"]) == 2
