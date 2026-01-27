"""
Paper-parallel test for envs/lever_game_env.py (LeverGameEnv).

CommNet (NIPS 2016) Lever task expectations:
- Pool of N agent IDs.
- Each episode samples m active agents uniformly without replacement.
- Each active agent observes ONLY its own identity (one-hot over N).
- Each active agent selects a lever.
- Shared reward = (# distinct levers) / m (or / num_levers in general).
- Episode ends after one step. (Stateless across episodes.)
- A deterministic "optimal" assignment exists by sorting IDs and assigning ranks to levers,
  which achieves reward 1.0 when num_levers == m.

This file tests those properties against your LeverGameEnv implementation.
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


def _assert_close(
    a: float, b: float, msg: str, atol: float = 1e-8, rtol: float = 1e-8
) -> None:
    if not math.isclose(a, b, abs_tol=atol, rel_tol=rtol):
        raise AssertionError(f"{msg} (got {a}, expected {b}, atol={atol}, rtol={rtol})")


def _decode_ids_from_onehot(obs: np.ndarray) -> np.ndarray:
    """
    Given obs shape [m, N] (one-hot per agent),
    returns inferred ids shape [m].
    """
    _assert(obs.ndim == 2, f"obs must be 2D, got {obs.ndim}D")
    ids = obs.argmax(axis=1).astype(np.int64)
    return ids


def test_reset_observation_shape_and_onehot_content() -> None:
    N = 50
    m = 5
    env = LeverGameEnv(pool_size=N, active_agents=m, seed=0)

    obs = env.reset()
    _assert(obs.shape == (m, N), f"obs shape expected {(m, N)}, got {obs.shape}")
    _assert(env.obs_dim == N, "obs_dim should equal pool_size")
    _assert(env.action_dim == env.num_levers, "action_dim should equal num_levers")

    # Each row should be exactly one-hot
    row_sums = obs.sum(axis=1)
    _assert(np.allclose(row_sums, 1.0), f"each obs row must sum to 1, got {row_sums}")

    # Values should be only 0 or 1
    _assert(np.all((obs == 0.0) | (obs == 1.0)), "obs entries must be 0 or 1")

    # The set of sampled IDs should be unique (without replacement)
    ids = _decode_ids_from_onehot(obs)
    _assert(len(set(ids.tolist())) == m, f"sampled ids must be unique, got {ids}")
    _assert(
        np.all(ids >= 0) and np.all(ids < N),
        "sampled ids must be within [0, pool_size)",
    )


def test_step_is_single_step_and_returns_done_true() -> None:
    env = LeverGameEnv(pool_size=100, active_agents=5, seed=0)
    obs0 = env.reset()

    actions = np.zeros((env.active_agents,), dtype=np.int64)
    obs1, reward, done, info = env.step(actions)

    _assert(done is True, "lever task should terminate after one step (done=True)")
    _assert(
        isinstance(reward, (float, np.floating)),
        f"reward should be scalar float, got {type(reward)}",
    )
    _assert(
        "distinct" in info and "success" in info,
        "info must contain 'distinct' and 'success'",
    )

    # Step returns next episode's obs (your env auto-resets)
    _assert(
        obs1.shape == obs0.shape,
        "obs after step should have same shape as obs from reset",
    )
    _assert(
        env.current_ids is not None,
        "env.current_ids should be set after step() because it resets internally",
    )


def test_reward_matches_distinct_over_num_levers() -> None:
    # Choose a smaller setup so we can reason exactly.
    N = 20
    m = 4
    L = 4
    env = LeverGameEnv(pool_size=N, active_agents=m, num_levers=L, seed=0)
    env.reset()

    # Case 1: all pick same lever => distinct=1 => reward=1/L
    _, r1, done1, info1 = env.step(np.array([0, 0, 0, 0], dtype=np.int64))
    _assert(done1 is True, "done should be True after one step")
    _assert(info1["distinct"] == 1, f"expected distinct=1, got {info1['distinct']}")
    _assert_close(float(r1), 1.0 / L, "reward should be distinct/num_levers")

    # Case 2: all pick different levers => distinct=4 => reward=1.0 (since L=4)
    _, r2, done2, info2 = env.step(np.array([0, 1, 2, 3], dtype=np.int64))
    _assert(done2 is True, "done should be True after one step")
    _assert(info2["distinct"] == 4, f"expected distinct=4, got {info2['distinct']}")
    _assert_close(
        float(r2), 1.0, "reward should be 1.0 when all levers are distinct and L==m"
    )
    _assert(
        info2["success"] is True, "success should be True when distinct == num_levers"
    )


def test_optimal_actions_give_perfect_reward_when_num_levers_equals_active_agents() -> (
    None
):
    # This is the exact "supervised optimal mapping" the paper uses for Lever.
    env = LeverGameEnv(pool_size=50, active_agents=5, num_levers=5, seed=123)
    env.reset()

    actions = env.optimal_actions()
    _assert(
        actions.shape == (env.active_agents,),
        "optimal_actions must return shape (active_agents,)",
    )

    # Optimal assignment should produce all distinct levers when L==m
    _, reward, done, info = env.step(actions)
    _assert(done is True, "done should be True after one step")
    _assert(
        info["distinct"] == env.num_levers,
        "optimal_actions should achieve distinct == num_levers when L==m",
    )
    _assert(
        info["success"] is True,
        "success should be True for optimal assignment when L==m",
    )
    _assert_close(
        float(reward), 1.0, "optimal_actions should achieve reward 1.0 when L==m"
    )


def test_optimal_actions_are_deterministic_given_current_ids() -> None:
    env = LeverGameEnv(pool_size=50, active_agents=5, num_levers=5, seed=0)
    env.reset()

    a1 = env.optimal_actions()
    a2 = env.optimal_actions()
    _assert(
        np.array_equal(a1, a2),
        "optimal_actions should be deterministic for fixed current_ids",
    )


def test_sample_random_action_bounds_and_shape() -> None:
    env = LeverGameEnv(pool_size=30, active_agents=6, num_levers=4, seed=0)
    env.reset()

    a = env.sample_random_action()
    _assert(
        a.shape == (env.active_agents,),
        f"random actions must have shape {(env.active_agents,)}, got {a.shape}",
    )
    _assert(
        np.all(a >= 0) and np.all(a < env.action_dim),
        "random actions must be within action space",
    )


@dataclass(frozen=True)
class _TestCase:
    name: str
    fn: Callable[[], None]


TESTS: list[_TestCase] = [
    _TestCase(
        "reset: obs shape + one-hot + unique IDs",
        test_reset_observation_shape_and_onehot_content,
    ),
    _TestCase(
        "step: single-step termination", test_step_is_single_step_and_returns_done_true
    ),
    _TestCase(
        "reward: distinct/num_levers formula",
        test_reward_matches_distinct_over_num_levers,
    ),
    _TestCase(
        "optimal_actions: perfect reward when L==m",
        test_optimal_actions_give_perfect_reward_when_num_levers_equals_active_agents,
    ),
    _TestCase(
        "optimal_actions: deterministic per current_ids",
        test_optimal_actions_are_deterministic_given_current_ids,
    ),
    _TestCase(
        "sample_random_action: bounds + shape",
        test_sample_random_action_bounds_and_shape,
    ),
]


def main() -> int:
    print("LeverGameEnv paper-parallel logic test")
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
