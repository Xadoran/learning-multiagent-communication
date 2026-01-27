"""
Full logic test for envs/coop_nav_env.py (CooperativeNavEnv).

Goal: not just "it runs", but "does it behave logically to the paper's specifications?"
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
    # Counts how many pairs are within collision radius (strictly <, matching the env)
    n = positions.shape[0]
    c = 0
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(positions[i] - positions[j])) < collision_radius:
                c += 1
    return c


def _pick_actions_one_step(env: CooperativeNavEnv, mode: str) -> np.ndarray:
    """
    Pick a per-agent action using one-step lookahead:
      - mode='toward': pick action that minimizes distance to its goal
      - mode='away'  : pick action that maximizes distance to its goal
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


def test_observation_shape_and_content() -> None:
    env = CooperativeNavEnv(num_agents=4, seed=0)
    obs = env.reset()

    _assert(
        obs.shape == (4, env.obs_dim),
        f"obs shape should be (J, obs_dim), got {obs.shape}",
    )
    _assert(env.obs_dim == 2 + 2 + 2 * (4 - 1), "obs_dim formula mismatch")

    # The first 2 entries of each obs row are the agent's position.
    # The next 2 entries are the agent's goal.
    for i in range(env.num_agents):
        _assert_close(
            float(obs[i][0]), float(env.positions[i][0]), "obs position x mismatch"
        )
        _assert_close(
            float(obs[i][1]), float(env.positions[i][1]), "obs position y mismatch"
        )
        _assert_close(float(obs[i][2]), float(env.goals[i][0]), "obs goal x mismatch")
        _assert_close(float(obs[i][3]), float(env.goals[i][1]), "obs goal y mismatch")


def test_vision_masking() -> None:
    env = CooperativeNavEnv(num_agents=2, seed=0, vision_radius=0.1)
    env.reset()

    # Force agents far apart (> vision), so relative observations should be zeros.
    env.positions[:] = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    env.goals[:] = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    obs = env._get_obs()

    rel0 = obs[0][-2:]
    rel1 = obs[1][-2:]
    _assert(
        np.allclose(rel0, np.zeros(2)),
        f"agent0 rel should be zeros when out of vision, got {rel0}",
    )
    _assert(
        np.allclose(rel1, np.zeros(2)),
        f"agent1 rel should be zeros when out of vision, got {rel1}",
    )

    # Now put them within vision; rels should be non-zero and consistent with positions.
    env.positions[:] = np.array([[0.0, 0.0], [0.05, 0.0]], dtype=np.float32)
    obs = env._get_obs()
    rel0 = obs[0][-2:]
    rel1 = obs[1][-2:]
    _assert(np.linalg.norm(rel0) > 0, "agent0 rel should be non-zero when in vision")
    _assert(np.linalg.norm(rel1) > 0, "agent1 rel should be non-zero when in vision")
    _assert(
        np.allclose(rel0, env.positions[1] - env.positions[0]),
        "agent0 rel vector incorrect",
    )
    _assert(
        np.allclose(rel1, env.positions[0] - env.positions[1]),
        "agent1 rel vector incorrect",
    )


def test_progress_reward_logic() -> None:
    # Same initial state, compare one-step 'toward' vs 'away'
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
        "mean distance after 'toward' step should be smaller than after 'away' step",
    )
    _assert(
        r_tow > r_away,
        "reward after 'toward' should be greater than reward after 'away'",
    )


def test_reward_math_matches_formula() -> None:
    # Force a deterministic configuration and check reward exactly matches the formula in the env.
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

    # Custom positions/goals to control collisions and progress.
    # - Make agents 0 and 1 collide (distance 0.1 < 0.2)
    # - Agent 2 far away
    env.positions[:] = np.array([[0.0, 0.0], [0.1, 0.0], [0.9, 0.9]], dtype=np.float32)
    env.goals[:] = np.array([[0.5, 0.5], [0.5, 0.5], [0.0, 0.0]], dtype=np.float32)

    # Set last_mean_dist to current mean_dist so progress=0 (makes expected reward simple).
    d0 = np.linalg.norm(env.positions - env.goals, axis=1)
    env.last_mean_dist = float(np.mean(d0))

    # Take 'stay' step so positions don't change (still collision)
    _, reward, done, info = env.step(np.array([0, 0, 0], dtype=np.int64))

    mean_dist = float(np.mean(np.linalg.norm(env.positions - env.goals, axis=1)))
    progress = 0.0
    collision_pairs = _count_collision_pairs(env.positions, env.collision_radius)
    collision_pen = collision_pairs * env.collision_penalty
    success = bool(
        np.all(np.linalg.norm(env.positions - env.goals, axis=1) < env.goal_eps)
    )
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
        done == success or done is False,
        "done should only be True on success or time limit",
    )
    _assert_close(
        float(reward),
        float(expected),
        "reward does not match expected formula",
        atol=1e-6,
        rtol=1e-6,
    )


def test_success_and_done() -> None:
    env = CooperativeNavEnv(
        num_agents=3, seed=0, goal_eps=0.2, success_bonus=20.0, max_steps=10
    )
    env.reset()

    # Place agents within goal_eps of their goals => success True on next step (even if they stay).
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


def test_time_limit_done() -> None:
    env = CooperativeNavEnv(num_agents=2, seed=0, max_steps=3, success_bonus=20.0)
    env.reset()

    done = False
    for _ in range(3):
        _, _reward, done, _info = env.step(np.array([0, 0], dtype=np.int64))
    _assert(done is True, "done should be True when max_steps is reached")


def test_position_clipping_to_bounds() -> None:
    # Note: CooperativeNavEnv's observation code assumes 2+ agents (it concatenates relative vectors).
    # This test uses 2 agents and only moves agent 0.
    env = CooperativeNavEnv(num_agents=2, seed=0, step_size=10.0)  # huge step
    env.reset()

    env.positions[:] = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    # action 1 = up; should clip to y=1.0
    env.step(np.array([1, 0], dtype=np.int64))
    _assert_close(
        float(env.positions[0][1]), 1.0, "agent0 y should clip to upper bound 1.0"
    )

    # action 2 = down; huge down should clip to y=-1.0
    env.step(np.array([2, 0], dtype=np.int64))
    _assert_close(
        float(env.positions[0][1]), -1.0, "agent0 y should clip to lower bound -1.0"
    )


@dataclass(frozen=True)
class _TestCase:
    name: str
    fn: Callable[[], None]


TESTS: list[_TestCase] = [
    _TestCase("observation shape/content", test_observation_shape_and_content),
    _TestCase("vision masking", test_vision_masking),
    _TestCase("progress reward logic", test_progress_reward_logic),
    _TestCase("reward math matches formula", test_reward_math_matches_formula),
    _TestCase("success sets done + bonus", test_success_and_done),
    _TestCase("time limit sets done", test_time_limit_done),
    _TestCase("position clipping to bounds", test_position_clipping_to_bounds),
]


def main() -> int:
    print("CooperativeNavEnv full logic test")
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
