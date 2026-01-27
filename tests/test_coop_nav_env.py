"""
CommNet-style logic test for envs/coop_nav_env.py (CooperativeNavEnv).

Important: This env is NOT one of the CommNet (NIPS 2016) paper's official benchmarks.
So this test is "paper-parallel in spirit": it checks the kinds of properties CommNet
relies on (partial observability, permutation symmetry, shared reward, coordination pressure),
in addition to verifying the env's explicit mechanics.
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

from envs.coop_nav_env import CooperativeNavEnv  # noqa: E402


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def _assert_close(
    a: float, b: float, msg: str, atol: float = 1e-6, rtol: float = 1e-6
) -> None:
    if not math.isclose(a, b, abs_tol=atol, rel_tol=rtol):
        raise AssertionError(f"{msg} (got {a}, expected {b}, atol={atol}, rtol={rtol})")


def _count_collision_pairs(positions: np.ndarray, collision_radius: float) -> int:
    n = positions.shape[0]
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(positions[i] - positions[j])) < collision_radius:
                c += 1
    return c


def _pick_actions_one_step(env: CooperativeNavEnv, mode: str) -> np.ndarray:
    """
    One-step lookahead heuristic:
      - mode='toward': pick action that minimizes distance to goal
      - mode='away'  : pick action that maximizes distance to goal
    """
    _assert(mode in ("toward", "away"), f"Unknown mode: {mode}")
    A = env.ACTIONS  # [5,2]
    acts = []
    for i in range(env.num_agents):
        p = env.positions[i]
        g = env.goals[i]
        cand = np.clip(p + A * env.step_size, -1.0, 1.0)
        d = np.linalg.norm(cand - g, axis=1)
        acts.append(int(np.argmin(d) if mode == "toward" else np.argmax(d)))
    return np.array(acts, dtype=np.int64)


def _permute_env_state(env: CooperativeNavEnv, perm: np.ndarray) -> None:
    """
    Permute agents in-place: positions and goals get re-ordered by perm.
    After this, agent i in the new env corresponds to old agent perm[i].
    """
    env.positions = env.positions[perm].copy()
    env.goals = env.goals[perm].copy()


def _permute_obs(obs: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """
    Permute obs rows by perm.
    """
    return obs[perm].copy()


def _permute_actions(actions: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """
    If env2's agent i corresponds to env1's agent perm[i],
    then to apply the same physical joint action:
        actions_env2[i] = actions_env1[perm[i]]
    """
    return actions[perm].copy()


def test_observation_shape_and_basic_content() -> None:
    env = CooperativeNavEnv(num_agents=4, seed=0)
    obs = env.reset()

    _assert(
        obs.shape == (4, env.obs_dim),
        f"obs shape should be (J, obs_dim), got {obs.shape}",
    )
    _assert(env.obs_dim == 2 + 2 + 2 * (4 - 1), "obs_dim formula mismatch")

    # The first 2 entries are position, next 2 are goal.
    for i in range(env.num_agents):
        _assert_close(
            float(obs[i][0]), float(env.positions[i][0]), "obs position x mismatch"
        )
        _assert_close(
            float(obs[i][1]), float(env.positions[i][1]), "obs position y mismatch"
        )
        _assert_close(float(obs[i][2]), float(env.goals[i][0]), "obs goal x mismatch")
        _assert_close(float(obs[i][3]), float(env.goals[i][1]), "obs goal y mismatch")


def test_vision_masking_contract() -> None:
    env = CooperativeNavEnv(num_agents=2, seed=0, vision_radius=0.1)
    env.reset()

    # Far apart => rel vectors are zero
    env.positions[:] = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    env.goals[:] = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    obs = env._get_obs()
    _assert(
        np.allclose(obs[0][-2:], 0.0),
        f"agent0 rel should be zero when out of vision, got {obs[0][-2:]}",
    )
    _assert(
        np.allclose(obs[1][-2:], 0.0),
        f"agent1 rel should be zero when out of vision, got {obs[1][-2:]}",
    )

    # Near => rel vectors match positions
    env.positions[:] = np.array([[0.0, 0.0], [0.05, 0.0]], dtype=np.float32)
    obs = env._get_obs()
    _assert(
        np.allclose(obs[0][-2:], env.positions[1] - env.positions[0]),
        "agent0 rel vector incorrect",
    )
    _assert(
        np.allclose(obs[1][-2:], env.positions[0] - env.positions[1]),
        "agent1 rel vector incorrect",
    )


def test_shared_reward_is_scalar_and_same_for_all_agents() -> None:
    env = CooperativeNavEnv(num_agents=3, seed=0)
    env.reset()
    obs, reward, done, info = env.step(np.array([0, 0, 0], dtype=np.int64))

    _assert(
        isinstance(reward, (float, np.floating)),
        f"reward should be scalar float, got {type(reward)}",
    )
    _assert(
        isinstance(done, (bool, np.bool_)), f"done should be bool, got {type(done)}"
    )
    _assert(
        "mean_dist" in info and "success" in info,
        "info must contain mean_dist and success keys",
    )


def test_progress_reward_logic_toward_vs_away() -> None:
    # Same init, compare one-step 'toward' vs 'away'
    seed = 123
    env_tow = CooperativeNavEnv(num_agents=3, seed=seed)
    env_away = CooperativeNavEnv(num_agents=3, seed=seed)
    env_tow.reset()
    env_away.reset()

    a_tow = _pick_actions_one_step(env_tow, "toward")
    a_away = _pick_actions_one_step(env_away, "away")

    _, r_tow, _, info_tow = env_tow.step(a_tow)
    _, r_away, _, info_away = env_away.step(a_away)

    _assert(
        info_tow["mean_dist"] < info_away["mean_dist"],
        "toward should reduce mean distance vs away",
    )
    _assert(r_tow > r_away, "toward should yield higher reward than away")


def test_reward_math_matches_env_formula_exactly() -> None:
    env = CooperativeNavEnv(
        num_agents=3,
        seed=0,
        distance_weight=0.7,
        progress_scale=2.0,
        time_penalty=0.1,
        collision_radius=0.2,
        collision_penalty=0.5,
        success_bonus=20.0,
        goal_eps=0.05,
        max_steps=40,
    )
    env.reset()

    # Deterministic setup:
    # - agents 0 and 1 collide (distance 0.1 < 0.2)
    env.positions[:] = np.array([[0.0, 0.0], [0.1, 0.0], [0.9, 0.9]], dtype=np.float32)
    env.goals[:] = np.array([[0.5, 0.5], [0.5, 0.5], [0.0, 0.0]], dtype=np.float32)

    # Make progress = 0 by setting last_mean_dist to the current mean distance
    d0 = np.linalg.norm(env.positions - env.goals, axis=1)
    env.last_mean_dist = float(np.mean(d0))

    # Step with 'stay'
    _, reward, done, info = env.step(np.array([0, 0, 0], dtype=np.int64))

    dists = np.linalg.norm(env.positions - env.goals, axis=1)
    mean_dist = float(np.mean(dists))
    progress = 0.0

    collision_pairs = _count_collision_pairs(env.positions, env.collision_radius)
    collision_pen = collision_pairs * env.collision_penalty

    success = bool(np.all(dists < env.goal_eps))
    expected = (
        -env.distance_weight * mean_dist
        + env.progress_scale * progress
        - collision_pen
        - env.time_penalty
        + (env.success_bonus if success else 0.0)
    )

    _assert_close(float(info["mean_dist"]), mean_dist, "info['mean_dist'] mismatch")
    _assert(info["collisions"] == (collision_pairs > 0), "info['collisions'] mismatch")
    _assert(
        done == (success or env.t >= env.max_steps),
        "done should be true only on success or time limit",
    )
    _assert_close(
        float(reward),
        float(expected),
        "reward mismatch vs formula",
        atol=1e-6,
        rtol=1e-6,
    )


def test_success_sets_done_and_bonus_applies() -> None:
    env = CooperativeNavEnv(
        num_agents=3, seed=0, goal_eps=0.2, success_bonus=20.0, max_steps=10
    )
    env.reset()

    env.goals[:] = np.array([[0.0, 0.0]] * 3, dtype=np.float32)
    env.positions[:] = np.array(
        [[0.0, 0.0], [0.05, 0.0], [0.0, 0.05]], dtype=np.float32
    )
    env.last_mean_dist = float(
        np.mean(np.linalg.norm(env.positions - env.goals, axis=1))
    )

    _, reward, done, info = env.step(np.array([0, 0, 0], dtype=np.int64))

    _assert(
        info["success"] is True,
        "success should be True when all agents are within goal_eps",
    )
    _assert(done is True, "done should be True on success")
    _assert(reward > 0.0, "reward should be positive when success bonus is applied")


def test_time_limit_sets_done() -> None:
    env = CooperativeNavEnv(num_agents=2, seed=0, max_steps=3, success_bonus=20.0)
    env.reset()

    done = False
    for _ in range(3):
        _, _reward, done, _info = env.step(np.array([0, 0], dtype=np.int64))

    _assert(
        env.t == env.max_steps,
        "env.t should equal max_steps after max_steps transitions",
    )
    _assert(done is True, "done should be True when max_steps is reached")


def test_position_clipping_to_bounds() -> None:
    env = CooperativeNavEnv(num_agents=2, seed=0, step_size=10.0)
    env.reset()

    env.positions[:] = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)

    # up => y clips to 1.0
    env.step(np.array([1, 0], dtype=np.int64))
    _assert_close(float(env.positions[0][1]), 1.0, "agent0 y should clip to +1.0")

    # down => y clips to -1.0
    env.step(np.array([2, 0], dtype=np.int64))
    _assert_close(float(env.positions[0][1]), -1.0, "agent0 y should clip to -1.0")


# ----------------------------
# CommNet-relevant (paper-style) properties
# ----------------------------


def test_agent_permutation_equivariance_of_dynamics_and_reward() -> None:
    """
    Agents are homogeneous; swapping agent indices should not change
    the physics or shared reward, only indexing.

    We test:
      - Permute agents (positions+goals) in env2
      - Permute the joint action consistently
      - Next positions should be permuted, reward/done identical
    """
    env1 = CooperativeNavEnv(num_agents=4, seed=42)
    env2 = CooperativeNavEnv(num_agents=4, seed=42)

    env1.reset()
    env2.reset()

    perm = np.array([2, 0, 3, 1], dtype=np.int64)

    # Permute env2's state to match env1 under reindexing
    _permute_env_state(env2, perm)

    # Pick action for env1, map it to env2 consistently
    a1 = env1.sample_random_action()
    a2 = _permute_actions(a1, perm)

    # Step both
    _obs1n, r1, d1, _ = env1.step(a1)
    _obs2n, r2, d2, _ = env2.step(a2)

    _assert_close(
        float(r1), float(r2), "reward should be invariant to agent permutation"
    )
    _assert(bool(d1) == bool(d2), "done should be invariant to agent permutation")

    # Positions/goals should match under the same reindexing
    for i_new in range(env2.num_agents):
        i_old = perm[i_new]
        _assert(
            np.allclose(env2.positions[i_new], env1.positions[i_old]),
            "next positions not permutation-consistent",
        )
        _assert(
            np.allclose(env2.goals[i_new], env1.goals[i_old]),
            "goals not permutation-consistent",
        )


def test_partial_observability_information_bottleneck() -> None:
    """
    CommNet paper relies on partial observability; here, vision_radius induces it.
    Test: when vision_radius is tiny and agents are separated, their observations about others are zero.
    """
    env = CooperativeNavEnv(num_agents=3, seed=0, vision_radius=1e-6)
    env.reset()

    # Force separation
    env.positions[:] = np.array(
        [[-0.9, -0.9], [0.0, 0.0], [0.9, 0.9]], dtype=np.float32
    )
    env.goals[:] = np.array([[0.0, 0.0]] * 3, dtype=np.float32)
    obs = env._get_obs()

    # Each agent's "others" part should be all zeros due to tiny vision radius
    for i in range(env.num_agents):
        others = obs[i][4:]  # after pos(2)+goal(2)
        _assert(
            np.allclose(others, 0.0),
            f"agent {i} should see no others when vision_radius ~ 0",
        )


@dataclass(frozen=True)
class _TestCase:
    name: str
    fn: Callable[[], None]


TESTS: list[_TestCase] = [
    _TestCase(
        "observation shape/basic content", test_observation_shape_and_basic_content
    ),
    _TestCase("vision masking contract", test_vision_masking_contract),
    _TestCase(
        "shared reward scalar", test_shared_reward_is_scalar_and_same_for_all_agents
    ),
    _TestCase(
        "progress reward toward vs away", test_progress_reward_logic_toward_vs_away
    ),
    _TestCase(
        "reward math matches formula", test_reward_math_matches_env_formula_exactly
    ),
    _TestCase("success sets done + bonus", test_success_sets_done_and_bonus_applies),
    _TestCase("time limit sets done", test_time_limit_sets_done),
    _TestCase("position clipping", test_position_clipping_to_bounds),
    _TestCase(
        "agent permutation equivariance",
        test_agent_permutation_equivariance_of_dynamics_and_reward,
    ),
    _TestCase(
        "partial observability bottleneck",
        test_partial_observability_information_bottleneck,
    ),
]


def main() -> int:
    print("CooperativeNavEnv CommNet-style logic test")
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
