"""Gymnasium and PettingZoo adapters over the same simultaneous Rust stepping API."""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from pettingzoo import ParallelEnv
from ._native import Batch, OBS_DIM


def scenario_config(scenario):
    if scenario is None:
        return {"vehicles": [{"id": "car_0", "controller": "external"}]}
    if isinstance(scenario, (str, Path)):
        return json.loads(Path(scenario).read_text())
    return deepcopy(scenario)


def _action(action):
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (2,) or not np.isfinite(value).all() or np.any(np.abs(value) > 1):
        raise ValueError("actions must have shape (2,) and finite values in [-1, 1]")
    return value


class DrivingEnv(gym.Env):
    """One external vehicle with optional native or learned background traffic."""
    metadata = {"render_modes": []}

    def __init__(self, scenario=None, agent_id=None):
        self.config = scenario_config(scenario)
        vehicles = self.config.setdefault("vehicles", [{"id": "car_0", "controller": "external"}])
        external = [i for i, v in enumerate(vehicles) if v.get("controller", "native") == "external"]
        if agent_id is not None:
            self.slot = next(i for i, v in enumerate(vehicles) if v["id"] == agent_id)
            vehicles[self.slot]["controller"] = "external"
            external = [i for i, v in enumerate(vehicles) if v.get("controller") == "external"]
        if len(external) != 1:
            raise ValueError("DrivingEnv requires exactly one external vehicle")
        self.slot = external[0]
        self.action_space = Box(-1., 1., (2,), np.float32)
        self.observation_space = Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.batch = Batch(json.dumps(self.config), [0])
        self.done = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        actual_seed = int(self.np_random.integers(0, 2**63)) if seed is None else seed
        result = self.batch.reset([actual_seed])
        self.done = False
        return result["observations"][0, self.slot].copy(), {"seed": actual_seed}

    def step(self, action):
        if self.done:
            raise RuntimeError("episode finished; call reset")
        actions = np.zeros((1, len(self.config["vehicles"]), 2), dtype=np.float32)
        actions[0, self.slot] = _action(action)
        r = self.batch.step(actions)
        obs = r["observations"][0, self.slot].copy()
        terminated, truncated = bool(r["terminated"][0, self.slot]), bool(r["truncated"][0, self.slot])
        self.done = terminated or truncated
        info = {"reason": r["reasons"][0][self.slot]}
        if self.done:
            info["final_observation"] = obs.copy()
        return obs, float(r["rewards"][0, self.slot]), terminated, truncated, info


class ParallelDrivingEnv(ParallelEnv):
    """Stable agent IDs; finished agents leave the API but their bodies remain."""
    metadata = {"name": "ml_cars_v0", "render_modes": [], "is_parallelizable": True}

    def __init__(self, scenario=None):
        self.config = scenario_config(scenario)
        self.slots = {v["id"]: i for i, v in enumerate(self.config["vehicles"]) if v.get("controller", "native") == "external"}
        if not self.slots:
            raise ValueError("ParallelDrivingEnv requires at least one external vehicle")
        self.possible_agents = list(self.slots)
        self.agents = []
        self.observation_spaces = {a: Box(-np.inf, np.inf, (OBS_DIM,), np.float32) for a in self.possible_agents}
        self.action_spaces = {a: Box(-1., 1., (2,), np.float32) for a in self.possible_agents}
        self.batch = Batch(json.dumps(self.config), [0])
        self.rng = np.random.default_rng()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        actual_seed = int(self.rng.integers(0, 2**63)) if seed is None else seed
        r = self.batch.reset([actual_seed])
        self.agents = self.possible_agents.copy()
        return ({a: r["observations"][0, self.slots[a]].copy() for a in self.agents}, {a: {"seed": actual_seed} for a in self.agents})

    def step(self, actions):
        if set(actions) != set(self.agents):
            raise ValueError("provide exactly one action per active external agent")
        if not self.agents:
            return {}, {}, {}, {}, {}
        array = np.zeros((1, len(self.config["vehicles"]), 2), dtype=np.float32)
        for a in self.agents:
            array[0, self.slots[a]] = _action(actions[a])
        r = self.batch.step(array)
        obs, rewards, terminated, truncated, infos = {}, {}, {}, {}, {}
        for a in self.agents:
            i = self.slots[a]
            obs[a] = r["observations"][0, i].copy()
            rewards[a] = float(r["rewards"][0, i])
            terminated[a], truncated[a] = bool(r["terminated"][0, i]), bool(r["truncated"][0, i])
            infos[a] = {"reason": r["reasons"][0][i]}
            if terminated[a] or truncated[a]:
                infos[a]["final_observation"] = obs[a].copy()
        self.agents = [a for a in self.agents if not terminated[a] and not truncated[a]]
        return obs, rewards, terminated, truncated, infos

    def close(self):
        pass
