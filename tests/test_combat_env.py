"""
Paper-parallel test for envs/combat_env.py (CombatEnv).

CommNet (NIPS 2016) Combat expectations:
- 15x15 grid, m agents per team
- Actions: noop + 4 moves + m attacks
- Reward: -0.1 * (total enemy HP), with terminal -1 on lose/draw
- Shared observation from local 3x3 neighborhood
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

from envs.combat_env import CombatEnv  # noqa: E402


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def _assert_close(
    a: float, b: float, msg: str, atol: float = 1e-6, rtol: float = 1e-6
) -> None:
    if not math.isclose(a, b, abs_tol=atol, rel_tol=rtol):
        raise AssertionError(f"{msg} (got {a}, expected {b}, atol={atol}, rtol={rtol})")


def test_reset_shapes_and_unique_positions() -> None:
    env = CombatEnv(grid_size=15, m=4, seed=0)
    obs = env.reset()
    _assert(obs.shape == (env.m, env.obs_dim), "obs shape mismatch")
    _assert(env.action_dim == 1 + 4 + env.m, "action_dim formula mismatch")

    # Ensure no two agents occupy the same cell at reset
    occupied = set()
    for i in range(env.m):
        occupied.add(tuple(env.model_pos[i]))
        occupied.add(tuple(env.bot_pos[i]))
    _assert(len(occupied) == 2 * env.m, "agents should spawn in unique cells")


def test_sample_random_action_shape_and_bounds() -> None:
    env = CombatEnv(m=3, seed=0)
    a = env.sample_random_action()
    _assert(a.shape == (env.m,), "random action shape mismatch")
    _assert(np.all(a >= 0) and np.all(a < env.action_dim), "actions out of bounds")


def test_step_returns_scalar_reward_and_info_keys() -> None:
    env = CombatEnv(m=3, seed=0)
    env.reset()
    a = env.sample_random_action()
    _obs, reward, done, info = env.step(a)

    _assert(isinstance(reward, (float, np.floating)), "reward should be scalar float")
    _assert(isinstance(done, (bool, np.bool_)), "done should be bool")
    for key in ["outcome", "model_alive", "bot_alive", "enemy_total_hp", "success"]:
        _assert(key in info, f"info missing key: {key}")


def test_reward_formula_matches_enemy_hp() -> None:
    env = CombatEnv(m=3, seed=0)
    env.reset()
    env.bot_hp[:] = np.array([3, 2, 1], dtype=np.int32)
    expected = -0.1 * float(np.sum(env.bot_hp))
    _assert_close(env._step_reward(), expected, "step reward mismatch")


def test_bot_policy_attacks_when_in_range() -> None:
    env = CombatEnv(m=1, seed=0)
    env.reset()
    env.model_pos[0] = np.array([5, 6], dtype=np.int32)
    env.bot_pos[0] = np.array([5, 5], dtype=np.int32)
    env.bot_cd[0] = 0

    actions = env._bot_policy_actions()
    _assert(actions.shape == (1,), "bot actions shape mismatch")
    _assert(actions[0] == 5, "bot should attack target 0 when in range")


def test_done_when_enemy_eliminated() -> None:
    env = CombatEnv(m=2, seed=0)
    env.reset()
    env.bot_hp[:] = 0
    done, outcome = env._check_done()
    _assert(done is True, "done should be True when all enemies dead")
    _assert(outcome == "win", "outcome should be win when enemies dead")


@dataclass(frozen=True)
class _TestCase:
    name: str
    fn: Callable[[], None]


TESTS: list[_TestCase] = [
    _TestCase("reset: shapes + unique positions", test_reset_shapes_and_unique_positions),
    _TestCase("random action: shape + bounds", test_sample_random_action_shape_and_bounds),
    _TestCase("step: reward scalar + info keys", test_step_returns_scalar_reward_and_info_keys),
    _TestCase("reward: matches enemy HP formula", test_reward_formula_matches_enemy_hp),
    _TestCase("bot policy: attack when in range", test_bot_policy_attacks_when_in_range),
    _TestCase("done: win when enemies eliminated", test_done_when_enemy_eliminated),
]


def main() -> int:
    print("CombatEnv paper-parallel logic test")
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
