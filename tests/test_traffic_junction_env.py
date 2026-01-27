"""
Paper-parallel test for envs/traffic_junction_env.py (TrafficJunctionEnv).

CommNet (NIPS 2016) Traffic Junction expectations:
- 14x14 grid, 3x3 local neighborhood (vision_range=1)
- New cars arrive stochastically; fixed number of slots (Nmax)
- Actions: BRAKE or GAS
- Reward: r(t) = C_t * r_coll + sum_i tau_i * r_time
- Episode ends after max_steps; failure if any collision occurred
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

import numpy as np

# Ensure repo root is on sys.path so `from envs...` works when running from tests/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.traffic_junction_env import TrafficJunctionEnv  # noqa: E402


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def _assert_close(
    a: float, b: float, msg: str, atol: float = 1e-6, rtol: float = 1e-6
) -> None:
    if not math.isclose(a, b, abs_tol=atol, rel_tol=rtol):
        raise AssertionError(f"{msg} (got {a}, expected {b}, atol={atol}, rtol={rtol})")


def test_reset_empty_obs_all_zero() -> None:
    env = TrafficJunctionEnv(
        grid_size=14, nmax=6, p_arrive=0.0, max_steps=5, vision_range=1, seed=0
    )
    obs = env.reset()
    _assert(obs.shape == (env.nmax, env.obs_dim), "obs shape mismatch")
    _assert(np.allclose(obs, 0.0), "obs should be all-zero when no cars present")


def test_sample_random_action_shape_and_bounds() -> None:
    env = TrafficJunctionEnv(grid_size=14, nmax=4, seed=0)
    a = env.sample_random_action()
    _assert(a.shape == (env.nmax,), "random action shape mismatch")
    _assert(np.all(a >= 0) and np.all(a < env.action_dim), "actions out of bounds")


def test_reward_matches_collision_and_time_formula() -> None:
    env = TrafficJunctionEnv(
        grid_size=14,
        nmax=3,
        p_arrive=0.0,
        max_steps=10,
        r_coll=-10.0,
        r_time=-0.01,
        vision_range=1,
        seed=0,
    )
    env.reset()

    # Set two cars on the same cell to force one collision pair.
    path = env._routes[("N", 0)]
    pos = path[0]
    env.present[:] = False
    env.present[0] = True
    env.present[1] = True
    env.pos[0] = pos
    env.pos[1] = pos
    env.route_id[0] = 0
    env.route_id[1] = 0
    env.route_idx[0] = 0
    env.route_idx[1] = 0
    env.tau[0] = 2
    env.tau[1] = 4
    env._set_route_key_for_slot(0, "N")
    env._set_route_key_for_slot(1, "N")

    # BRAKE for all
    actions = np.zeros((env.nmax,), dtype=np.int64)
    _obs, reward, done, info = env.step(actions)

    expected_collisions = 1  # two cars in one cell -> 1 pair
    expected_time = (2 + 4) * env.r_time
    expected_reward = expected_collisions * env.r_coll + expected_time

    _assert(info["collisions_t"] == expected_collisions, "collision count mismatch")
    _assert(info["failure"] is True, "failure should be True after collision")
    _assert(done is False, "done should be False before max_steps")
    _assert_close(reward, expected_reward, "reward mismatch vs formula")


def test_success_after_horizon_no_collision() -> None:
    env = TrafficJunctionEnv(
        grid_size=14, nmax=5, p_arrive=0.0, max_steps=3, vision_range=1, seed=0
    )
    env.reset()
    done = False
    info = {}
    for _ in range(env.max_steps):
        _obs, _reward, done, info = env.step(np.zeros((env.nmax,), dtype=np.int64))

    _assert(done is True, "done should be True after max_steps")
    _assert(info["failure"] is False, "failure should be False when no collisions")
    _assert(info["success"] is True, "success should be True when no collisions")


def test_obs_present_flag_and_tau_normalization() -> None:
    env = TrafficJunctionEnv(
        grid_size=14, nmax=4, p_arrive=0.0, max_steps=20, vision_range=1, seed=0
    )
    env.reset()
    env.present[:] = False
    env.present[2] = True
    env.route_id[2] = 1
    env.route_idx[2] = 0
    env.tau[2] = 10
    env._set_route_key_for_slot(2, "E")
    path = env._routes[("E", 1)]
    env.pos[2] = path[0]

    obs = env._get_obs()
    row = obs[2]

    # present flag is first element
    _assert_close(float(row[0]), 1.0, "present flag should be 1.0")

    # tau normalized scalar index
    tau_idx = 1 + env.id_dim + env.loc_dim + env.route_dim
    expected_tau = float(env.tau[2]) / float(env.max_steps)
    _assert_close(float(row[tau_idx]), expected_tau, "tau normalization mismatch")


@dataclass(frozen=True)
class _TestCase:
    name: str
    fn: Callable[[], None]


TESTS: list[_TestCase] = [
    _TestCase("reset: empty obs is all zero", test_reset_empty_obs_all_zero),
    _TestCase("random action: shape + bounds", test_sample_random_action_shape_and_bounds),
    _TestCase("reward: collision + time formula", test_reward_matches_collision_and_time_formula),
    _TestCase("success: no collision by horizon", test_success_after_horizon_no_collision),
    _TestCase("obs: present flag + tau normalized", test_obs_present_flag_and_tau_normalization),
]


def main() -> int:
    print("TrafficJunctionEnv paper-parallel logic test")
    print(f"- Running {len(TESTS)} checks\n")

    failures: list[str] = []
    for t in TESTS:
        try:
            t.fn()
            print(f"[PASS] {t.name}")
        except Exception as e:
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
