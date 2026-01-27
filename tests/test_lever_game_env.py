"""
Full logic test for envs/lever_game_env.py (LeverGameEnv).

Goal: not just "it runs", but "does it behave logically?".
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np

# Ensure repo root is on sys.path so `from envs...` works when running from tests/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.lever_game_env import LeverGameEnv  # noqa: E402


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def _assert_close(a: float, b: float, msg: str, atol: float = 1e-6, rtol: float = 1e-6) -> None:
    if not math.isclose(a, b, abs_tol=atol, rel_tol=rtol):
        raise AssertionError(f"{msg} (got {a}, expected {b}, atol={atol}, rtol={rtol})")


def _is_one_hot_rowwise(obs: np.ndarray) -> bool:
    # Each row should have exactly one 1.0 and the rest 0.0.
    if obs.ndim != 2:
        return False
    row_sums = obs.sum(axis=1)
    if not np.allclose(row_sums, 1.0):
        return False
    # Values should be in {0,1} (float32)
    if not np.all((obs == 0.0) | (obs == 1.0)):
        return False
    return True


def test_reset_shapes_and_one_hot() -> None:
    env = LeverGameEnv(pool_size=50, active_agents=5, seed=0)
    obs = env.reset()
    _assert(obs.shape == (env.active_agents, env.pool_size), f"obs shape mismatch: {obs.shape}")
    _assert(env.obs_dim == env.pool_size, "obs_dim should equal pool_size")
    _assert(env.action_dim == env.num_levers, "action_dim should equal num_levers")
    _assert(_is_one_hot_rowwise(obs), "reset() observation should be one-hot per agent")

    # current_ids should be set and unique
    _assert(env.current_ids is not None, "current_ids should be set after reset()")
    _assert(len(set(env.current_ids.tolist())) == env.active_agents, "current_ids should be unique (sampled w/o replacement)")
    _assert(np.all((0 <= env.current_ids) & (env.current_ids < env.pool_size)), "current_ids must be in [0, pool_size)")


def test_num_levers_default() -> None:
    env = LeverGameEnv(pool_size=100, active_agents=7, num_levers=None, seed=0)
    _assert(env.num_levers == env.active_agents, "num_levers should default to active_agents when None")
    _assert(env.action_dim == env.active_agents, "action_dim should match default num_levers")


def test_step_is_single_turn_and_resets_obs() -> None:
    env = LeverGameEnv(pool_size=30, active_agents=4, seed=0)
    obs0 = env.reset()
    ids0 = env.current_ids.copy()

    actions = np.zeros((env.active_agents,), dtype=np.int64)
    obs1, reward, done, info = env.step(actions)

    _assert(done is True, "lever game should end after one step (done=True)")
    _assert(isinstance(info, dict) and "distinct" in info and "success" in info, "info should contain distinct and success")
    _assert(obs1.shape == obs0.shape, "obs shape should remain consistent after step()")
    _assert(_is_one_hot_rowwise(obs1), "step() should return a fresh one-hot observation (it calls reset())")

    # Because step() calls reset(), current_ids should usually change (not guaranteed, but overwhelmingly likely).
    ids1 = env.current_ids.copy()
    _assert(ids1 is not None and ids1.shape == ids0.shape, "current_ids should be updated after step()")

    # Reward should be in [1/num_levers, 1]
    _assert(0.0 <= reward <= 1.0, "reward should be in [0,1]")
    _assert(reward >= 1.0 / env.num_levers, "minimum distinct is 1, so reward >= 1/num_levers")


def test_reward_formula_matches_distinct_count() -> None:
    env = LeverGameEnv(pool_size=50, active_agents=5, num_levers=5, seed=0)
    env.reset()

    # 1) all same lever -> distinct=1
    a = np.array([0, 0, 0, 0, 0], dtype=np.int64)
    _obs, r, done, info = env.step(a)
    _assert(done is True, "done should be True")
    _assert(info["distinct"] == 1, "distinct should be 1 when all pick same lever")
    _assert_close(float(r), 1.0 / env.num_levers, "reward should equal distinct/num_levers")

    # 2) all different levers -> distinct=num_levers -> success True, reward=1
    env.reset()
    a = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    _obs, r, done, info = env.step(a)
    _assert(info["distinct"] == env.num_levers, "distinct should equal num_levers when all different")
    _assert(info["success"] is True, "success should be True when distinct == num_levers")
    _assert_close(float(r), 1.0, "reward should be 1.0 when distinct == num_levers")


def test_optimal_actions_produces_success() -> None:
    # With num_levers == active_agents, the provided optimal_actions() should produce a perfect assignment.
    env = LeverGameEnv(pool_size=50, active_agents=5, num_levers=5, seed=0)
    env.reset()
    a = env.optimal_actions()
    _obs, r, done, info = env.step(a)
    _assert(done is True, "done should be True")
    _assert(info["success"] is True, "optimal_actions should achieve success when num_levers == active_agents")
    _assert_close(float(r), 1.0, "optimal_actions should achieve reward 1.0 in this setting")


def test_seed_reproducibility_for_reset() -> None:
    # Two envs with same seed should sample the same first set of IDs.
    env1 = LeverGameEnv(pool_size=50, active_agents=5, seed=123)
    env2 = LeverGameEnv(pool_size=50, active_agents=5, seed=123)
    env1.reset()
    env2.reset()
    _assert(np.array_equal(env1.current_ids, env2.current_ids), "same seed should reproduce the same first reset() IDs")


@dataclass(frozen=True)
class _TestCase:
    name: str
    fn: Callable[[], None]


TESTS: list[_TestCase] = [
    _TestCase("reset shapes + one-hot + unique IDs", test_reset_shapes_and_one_hot),
    _TestCase("num_levers default behavior", test_num_levers_default),
    _TestCase("step is single-turn and returns new obs", test_step_is_single_turn_and_resets_obs),
    _TestCase("reward equals distinct/num_levers", test_reward_formula_matches_distinct_count),
    _TestCase("optimal_actions achieves success", test_optimal_actions_produces_success),
    _TestCase("seed reproducibility (first reset)", test_seed_reproducibility_for_reset),
]


def main() -> int:
    print("LeverGameEnv full logic test")
    print(f"- Running {len(TESTS)} checks\n")

    failures: list[str] = []
    for t in TESTS:
        try:
            t.fn()
            print(f"[PASS] {t.name}")
        except Exception as e:  # noqa: BLE001 - want a clean test report
            failures.append(f"{t.name}: {e}")
            print(f"[FAIL] {t.name}\n  -> {e}")

    if failures:
        print("\nSummary: FAIL")
        print(f"- {len(failures)} / {len(TESTS)} checks failed")
        return 1

    print("\nSummary: PASS (all checks succeeded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

