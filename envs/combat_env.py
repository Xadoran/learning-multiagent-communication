# envs/combat_env.py
import numpy as np


class CombatEnv:
    """
    Combat task environment (CommNet paper, NIPS 2016).

    - 15x15 grid
    - Two teams: model-controlled team (red) and bot team (blue)
    - Each team has m agents (default m=5)
    - Initial positions:
        * Pick a random "team center" uniformly in grid
        * Sample each agent uniformly in a 5x5 square around that center (clipped)
        * Ensure no two agents share the same cell initially

    - At each step, each agent can:
        * do nothing
        * move 1 cell in one of 4 directions
        * attack an enemy by specifying its ID j (m attack actions)
            - attack succeeds only if target is within firing range (surrounding 3x3 area)
            - after attacking, attacker enters cooldown for 1 step (cannot attack next step)

    - Health:
        * All agents start with 3 HP
        * Die when HP reaches 0 (removed from board)

    - Episode ends when:
        * one team eliminated (win/lose), OR
        * max_steps reached (draw)

    - Bot policy (hard-coded, as in paper):
        * If nearest enemy is within firing range -> attack it
        * Else approach nearest *visible* enemy within visual range
        * Shared vision for bots: a model agent is visible to ALL bots if it is
          within visual range of ANY bot.

    - Observations (fixed-size per model-controlled agent):
        Paper describes one-hot vectors {i, t, l, h, c} for agents:
          i: unique ID
          t: team ID
          l: location
          h: health points
          c: cooldown
        And each model agent sees other agents in its 3x3 visual range.

        For practicality with a fixed obs_dim, we encode:
          * self features: one-hot(id=m), one-hot(team=2), one-hot(loc=grid^2),
                           one-hot(hp=3), one-hot(cooldown=2)
          * up to 8 visible "other" agents (in 3x3 neighborhood, excluding self),
            each as a slot containing:
              - one-hot(id=2m)  (global ID among all agents)
              - one-hot(team=2)
              - one-hot(rel_pos=9)  (position within the 3x3 window)
              - one-hot(hp=3)
              - one-hot(cooldown=2)
            Remaining slots padded with zeros.

    - Reward (as described in paper):
        * At each step: r = -0.1 * (total enemy HP)
        * Terminal: if model team loses OR draw: additional -1
          (Win gives no extra reward; the shaping reward encourages attacking.)
    """

    # Moves: noop, up, down, left, right
    MOVE_DELTAS = np.array(
        [
            [0, 0],  # 0 noop
            [0, 1],  # 1 up
            [0, -1],  # 2 down
            [-1, 0],  # 3 left
            [1, 0],  # 4 right
        ],
        dtype=np.int32,
    )

    def __init__(
        self,
        grid_size: int = 15,
        m: int = 5,
        max_steps: int = 40,
        spawn_square: int = 5,  # 5x5 square around team center
        visual_range: int = 1,  # 3x3 visual range => chebyshev distance <= 1
        firing_range: int = 1,  # 3x3 firing range => chebyshev distance <= 1
        hp_max: int = 3,
        cooldown_steps: int = 1,  # cooldown duration after an attack
        seed: int | None = None,
    ):
        assert (
            spawn_square == 5
        ), "Paper uses a 5x5 spawn square; you can change, but then it's no longer 'paper-parallel'."
        self.grid_size = grid_size
        self.m = m
        self.max_steps = max_steps
        self.spawn_square = spawn_square
        self.visual_range = visual_range
        self.firing_range = firing_range
        self.hp_max = hp_max
        self.cooldown_steps = cooldown_steps
        self.rng = np.random.default_rng(seed)

        # Action space: noop + 4 moves + m attacks
        self.action_dim = 1 + 4 + self.m

        # Observation dims (fixed)
        self.self_id_dim = self.m
        self.team_dim = 2
        self.loc_dim = self.grid_size * self.grid_size
        self.hp_dim = self.hp_max
        self.cd_dim = 2

        self.other_slots = 8  # max other agents in 3x3 neighborhood excluding self
        self.other_id_dim = 2 * self.m
        self.relpos_dim = 9  # positions in 3x3 window

        self.self_feat_dim = (
            self.self_id_dim + self.team_dim + self.loc_dim + self.hp_dim + self.cd_dim
        )
        self.other_feat_dim = (
            self.other_id_dim
            + self.team_dim
            + self.relpos_dim
            + self.hp_dim
            + self.cd_dim
        )

        self.obs_dim = self.self_feat_dim + self.other_slots * self.other_feat_dim

        # State arrays
        # Model team indices: 0..m-1
        # Bot team indices: 0..m-1 (separate arrays)
        self.model_pos = np.zeros((self.m, 2), dtype=np.int32)
        self.bot_pos = np.zeros((self.m, 2), dtype=np.int32)

        self.model_hp = np.zeros((self.m,), dtype=np.int32)
        self.bot_hp = np.zeros((self.m,), dtype=np.int32)

        self.model_cd = np.zeros((self.m,), dtype=np.int32)  # cooldown timer
        self.bot_cd = np.zeros((self.m,), dtype=np.int32)

        self.t = 0

    # ----------------------------
    # Public API
    # ----------------------------
    def reset(self):
        self.t = 0
        self.model_hp[:] = self.hp_max
        self.bot_hp[:] = self.hp_max
        self.model_cd[:] = 0
        self.bot_cd[:] = 0

        # Sample team centers uniformly
        model_center = self.rng.integers(0, self.grid_size, size=2)
        bot_center = self.rng.integers(0, self.grid_size, size=2)

        # Spawn agents around centers in 5x5 square, avoid overlaps across all agents
        occupied = set()
        self.model_pos = self._spawn_team(model_center, occupied)
        self.bot_pos = self._spawn_team(bot_center, occupied)

        return self._get_obs()

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (
            self.m,
        ), f"Expected actions shape ({self.m},), got {actions.shape}"

        # 1) Model team acts (simultaneous-ish; resolve moves first, then attacks)
        model_move_targets, model_attacks = self._decode_actions(actions, team="model")

        # 2) Bot team chooses actions via hard-coded policy
        bot_actions = self._bot_policy_actions()
        bot_move_targets, bot_attacks = self._decode_actions(bot_actions, team="bot")

        # 3) Apply movements (with simple collision handling: agents can't move into occupied cells)
        #    Paper doesn't specify movement collisions; MazeBase typically prevents overlaps.
        self._apply_moves(model_move_targets, bot_move_targets)

        # 4) Apply attacks (model then bot, or simultaneous; paper doesn't specify ordering.
        #    We'll do simultaneous damage accumulation to be fair.)
        self._apply_attacks_simultaneous(model_attacks, bot_attacks)

        # 5) Update cooldown timers
        self.model_cd = np.maximum(self.model_cd - 1, 0)
        self.bot_cd = np.maximum(self.bot_cd - 1, 0)

        # 6) Check termination and compute reward
        self.t += 1
        done, outcome = self._check_done()

        reward = self._step_reward()
        if done and outcome in ("lose", "draw"):
            reward += -1.0

        obs = self._get_obs()
        info = {
            "outcome": outcome,  # "win" / "lose" / "draw" / "ongoing"
            "model_alive": int(np.sum(self.model_hp > 0)),
            "bot_alive": int(np.sum(self.bot_hp > 0)),
            "enemy_total_hp": int(np.sum(self.bot_hp[self.bot_hp > 0])),
            "success": outcome == "win",
        }
        return obs, reward, done, info

    def sample_random_action(self):
        return self.rng.integers(0, self.action_dim, size=(self.m,), dtype=np.int64)

    # ----------------------------
    # Spawning
    # ----------------------------
    def _spawn_team(self, center_xy: np.ndarray, occupied: set[tuple[int, int]]):
        half = self.spawn_square // 2  # 2 for 5x5
        positions = np.zeros((self.m, 2), dtype=np.int32)

        for i in range(self.m):
            for _ in range(10_000):
                dx = self.rng.integers(-half, half + 1)
                dy = self.rng.integers(-half, half + 1)
                x = int(np.clip(center_xy[0] + dx, 0, self.grid_size - 1))
                y = int(np.clip(center_xy[1] + dy, 0, self.grid_size - 1))
                if (x, y) not in occupied:
                    occupied.add((x, y))
                    positions[i] = [x, y]
                    break
            else:
                raise RuntimeError(
                    "Failed to spawn without collisions. Try a larger grid or smaller m."
                )
        return positions

    # ----------------------------
    # Action decoding
    # ----------------------------
    def _decode_actions(self, actions: np.ndarray, team: str):
        """
        Returns:
          move_targets: (m, 2) int32, desired positions (or current if noop/attack)
          attacks: list of (attacker_idx, target_idx) in enemy indexing [0..m-1]
        """
        if team == "model":
            pos = self.model_pos
            cd = self.model_cd
        else:
            pos = self.bot_pos
            cd = self.bot_cd

        move_targets = pos.copy()
        attacks = []

        for i, a in enumerate(actions.tolist()):
            if a <= 4:
                # noop or move
                delta = self.MOVE_DELTAS[a]
                tgt = pos[i] + delta
                tgt = np.clip(tgt, 0, self.grid_size - 1)
                move_targets[i] = tgt
            else:
                # attack action
                enemy_j = a - 5
                if 0 <= enemy_j < self.m:
                    # Only if not in cooldown
                    if cd[i] == 0:
                        attacks.append((i, enemy_j))
                        move_targets[i] = pos[i]  # attacks do not move
                        cd[i] = self.cooldown_steps  # set cooldown immediately
                    else:
                        move_targets[i] = pos[i]
                else:
                    move_targets[i] = pos[i]

        return move_targets, attacks

    # ----------------------------
    # Movement / collision
    # ----------------------------
    def _apply_moves(self, model_targets: np.ndarray, bot_targets: np.ndarray):
        """
        Prevent two agents from occupying the same cell.
        If a target cell is already occupied (by someone who stays or moves there),
        the move is cancelled (agent remains in place).
        """
        # Build current occupancy of alive agents
        occupied = set()
        for i in range(self.m):
            if self.model_hp[i] > 0:
                occupied.add(tuple(self.model_pos[i]))
            if self.bot_hp[i] > 0:
                occupied.add(tuple(self.bot_pos[i]))

        # Resolve moves in a randomized order to avoid bias
        order = self.rng.permutation(2 * self.m)
        for k in order:
            if k < self.m:
                # model agent k
                i = k
                if self.model_hp[i] <= 0:
                    continue
                cur = tuple(self.model_pos[i])
                tgt = tuple(model_targets[i])
                if tgt == cur:
                    continue
                # temporarily free current cell, attempt move
                occupied.remove(cur)
                if tgt in occupied:
                    # blocked, stay
                    occupied.add(cur)
                else:
                    self.model_pos[i] = np.array(tgt, dtype=np.int32)
                    occupied.add(tgt)
            else:
                # bot agent
                i = k - self.m
                if self.bot_hp[i] <= 0:
                    continue
                cur = tuple(self.bot_pos[i])
                tgt = tuple(bot_targets[i])
                if tgt == cur:
                    continue
                occupied.remove(cur)
                if tgt in occupied:
                    occupied.add(cur)
                else:
                    self.bot_pos[i] = np.array(tgt, dtype=np.int32)
                    occupied.add(tgt)

    # ----------------------------
    # Attacks
    # ----------------------------
    def _in_range(
        self, attacker_xy: np.ndarray, target_xy: np.ndarray, rng: int
    ) -> bool:
        # 3x3 area == chebyshev distance <= 1
        return (
            max(
                abs(int(attacker_xy[0] - target_xy[0])),
                abs(int(attacker_xy[1] - target_xy[1])),
            )
            <= rng
        )

    def _apply_attacks_simultaneous(self, model_attacks, bot_attacks):
        """
        Apply damage simultaneously:
          - each valid attack reduces target HP by 1
          - only if target alive and in firing range (3x3)
        """
        dmg_to_bot = np.zeros((self.m,), dtype=np.int32)
        dmg_to_model = np.zeros((self.m,), dtype=np.int32)

        # Model attacks bots
        for attacker_i, target_j in model_attacks:
            if self.model_hp[attacker_i] <= 0:
                continue
            if self.bot_hp[target_j] <= 0:
                continue
            if self._in_range(
                self.model_pos[attacker_i], self.bot_pos[target_j], self.firing_range
            ):
                dmg_to_bot[target_j] += 1

        # Bot attacks model
        for attacker_i, target_j in bot_attacks:
            if self.bot_hp[attacker_i] <= 0:
                continue
            if self.model_hp[target_j] <= 0:
                continue
            if self._in_range(
                self.bot_pos[attacker_i], self.model_pos[target_j], self.firing_range
            ):
                dmg_to_model[target_j] += 1

        # Apply damage
        self.bot_hp = np.maximum(self.bot_hp - dmg_to_bot, 0)
        self.model_hp = np.maximum(self.model_hp - dmg_to_model, 0)

    # ----------------------------
    # Bot policy (hard-coded)
    # ----------------------------
    def _bot_policy_actions(self) -> np.ndarray:
        """
        Bot policy (paper):
          - attack nearest enemy if within firing range
          - else approach nearest visible enemy within visual range
          - shared vision for bots: model agent is visible if within visual range of any bot
        """
        actions = np.zeros((self.m,), dtype=np.int64)

        # Determine which model agents are "visible to bots" (shared vision)
        visible_model = np.zeros((self.m,), dtype=bool)
        for bi in range(self.m):
            if self.bot_hp[bi] <= 0:
                continue
            for mi in range(self.m):
                if self.model_hp[mi] <= 0:
                    continue
                if self._in_range(
                    self.bot_pos[bi], self.model_pos[mi], self.visual_range
                ):
                    visible_model[mi] = True

        # Precompute visible targets positions
        visible_targets = [
            mi for mi in range(self.m) if visible_model[mi] and self.model_hp[mi] > 0
        ]

        for bi in range(self.m):
            if self.bot_hp[bi] <= 0:
                actions[bi] = 0
                continue

            # If in cooldown, can only move/noop
            can_attack = self.bot_cd[bi] == 0

            # Find nearest visible model agent (if any)
            target = None
            if visible_targets:
                dists = []
                for mi in visible_targets:
                    dx = int(self.model_pos[mi][0] - self.bot_pos[bi][0])
                    dy = int(self.model_pos[mi][1] - self.bot_pos[bi][1])
                    d = abs(dx) + abs(dy)
                    dists.append((d, mi))
                dists.sort(key=lambda x: x[0])
                target = dists[0][1]

            # Attack if possible and target within firing range
            if (
                target is not None
                and can_attack
                and self._in_range(
                    self.bot_pos[bi], self.model_pos[target], self.firing_range
                )
            ):
                actions[bi] = 5 + target  # attack target ID
                continue

            # Otherwise move toward target if exists; else noop
            if target is None:
                actions[bi] = 0
                continue

            # Move greedily to reduce Manhattan distance
            dx = int(self.model_pos[target][0] - self.bot_pos[bi][0])
            dy = int(self.model_pos[target][1] - self.bot_pos[bi][1])

            # Pick primary axis randomly if tie
            if abs(dx) > abs(dy):
                step = np.array([np.sign(dx), 0], dtype=np.int32)
            elif abs(dy) > abs(dx):
                step = np.array([0, np.sign(dy)], dtype=np.int32)
            else:
                if self.rng.random() < 0.5:
                    step = np.array([np.sign(dx), 0], dtype=np.int32)
                else:
                    step = np.array([0, np.sign(dy)], dtype=np.int32)

            # Convert step to action index
            if step[0] == 1:
                actions[bi] = 4  # right
            elif step[0] == -1:
                actions[bi] = 3  # left
            elif step[1] == 1:
                actions[bi] = 1  # up
            elif step[1] == -1:
                actions[bi] = 2  # down
            else:
                actions[bi] = 0

        return actions

    # ----------------------------
    # Reward / Done
    # ----------------------------
    def _check_done(self):
        model_alive = np.any(self.model_hp > 0)
        bot_alive = np.any(self.bot_hp > 0)

        if not model_alive and not bot_alive:
            return True, "draw"  # extremely unlikely
        if not bot_alive:
            return True, "win"
        if not model_alive:
            return True, "lose"
        if self.t >= self.max_steps:
            return True, "draw"
        return False, "ongoing"

    def _step_reward(self) -> float:
        # Paper shaping reward: -0.1 * total enemy HP (encourages attacking)
        enemy_total_hp = float(np.sum(self.bot_hp[self.bot_hp > 0]))
        return -0.1 * enemy_total_hp

    # ----------------------------
    # Observation encoding
    # ----------------------------
    def _one_hot(self, idx: int, dim: int) -> np.ndarray:
        v = np.zeros((dim,), dtype=np.float32)
        if 0 <= idx < dim:
            v[idx] = 1.0
        return v

    def _loc_index(self, xy: np.ndarray) -> int:
        x, y = int(xy[0]), int(xy[1])
        return y * self.grid_size + x

    def _relpos_index_3x3(self, dx: int, dy: int) -> int:
        # dx,dy in {-1,0,1}; map to 0..8 in row-major
        return (dy + 1) * 3 + (dx + 1)

    def _get_obs(self) -> np.ndarray:
        obs = []
        # Build a quick lookup of all alive agents in grid
        # global IDs: model 0..m-1, bot m..2m-1
        grid_agents = {}  # (x,y) -> list of (global_id, team, hp, cd)
        for i in range(self.m):
            if self.model_hp[i] > 0:
                key = tuple(self.model_pos[i])
                grid_agents.setdefault(key, []).append(
                    (i, 0, int(self.model_hp[i]), int(self.model_cd[i]))
                )
            if self.bot_hp[i] > 0:
                key = tuple(self.bot_pos[i])
                grid_agents.setdefault(key, []).append(
                    (self.m + i, 1, int(self.bot_hp[i]), int(self.bot_cd[i]))
                )

        for i in range(self.m):
            if self.model_hp[i] <= 0:
                # dead agents still need a fixed obs vector
                obs.append(np.zeros((self.obs_dim,), dtype=np.float32))
                continue

            # ---- self features
            self_id = self._one_hot(i, self.self_id_dim)
            self_team = self._one_hot(0, self.team_dim)  # model team = 0
            self_loc = self._one_hot(self._loc_index(self.model_pos[i]), self.loc_dim)
            self_hp = self._one_hot(
                int(self.model_hp[i]) - 1, self.hp_dim
            )  # hp in {1..3} -> {0..2}
            self_cd = self._one_hot(1 if self.model_cd[i] > 0 else 0, self.cd_dim)

            self_feat = np.concatenate(
                [self_id, self_team, self_loc, self_hp, self_cd], axis=0
            )

            # ---- other features: scan 3x3 neighborhood
            others = []
            x0, y0 = int(self.model_pos[i][0]), int(self.model_pos[i][1])
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    gx, gy = x0 + dx, y0 + dy
                    if not (0 <= gx < self.grid_size and 0 <= gy < self.grid_size):
                        continue
                    cell = (gx, gy)
                    if cell not in grid_agents:
                        continue
                    # If multiple agents somehow in same cell, include them all (truncate to slots later)
                    for gid, team, hp, cd in grid_agents[cell]:
                        # skip self
                        if gid == i:
                            continue
                        other_id = self._one_hot(gid, self.other_id_dim)
                        other_team = self._one_hot(team, self.team_dim)
                        relpos = self._one_hot(
                            self._relpos_index_3x3(dx, dy), self.relpos_dim
                        )
                        other_hp = (
                            self._one_hot(hp - 1, self.hp_dim)
                            if hp > 0
                            else np.zeros((self.hp_dim,), dtype=np.float32)
                        )
                        other_cd = self._one_hot(1 if cd > 0 else 0, self.cd_dim)
                        slot = np.concatenate(
                            [other_id, other_team, relpos, other_hp, other_cd], axis=0
                        )
                        others.append(slot)

            # Pad / truncate to fixed slots
            if len(others) < self.other_slots:
                pad = [
                    np.zeros((self.other_feat_dim,), dtype=np.float32)
                    for _ in range(self.other_slots - len(others))
                ]
                others.extend(pad)
            else:
                others = others[: self.other_slots]

            obs_i = np.concatenate([self_feat] + others, axis=0)
            obs.append(obs_i)

        return np.stack(obs, axis=0)
