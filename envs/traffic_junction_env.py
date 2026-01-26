import numpy as np


class TrafficJunctionEnv:
    """
    Traffic Junction coordination environment (CommNet paper).
    - Grid-based environment where agents (cars) must coordinate to cross intersections.
    - Agents spawn at different entry points and need to reach exit points.
    - Collision penalty for agents occupying the same cell.
    - Partial observability: agents observe local grid around them.
    - Shared reward: success bonus when all agents reach goals, minus collision penalties.
    """

    # Actions: stay, up, down, left, right
    ACTIONS = np.array(
        [
            [0, 0],   # stay
            [0, 1],   # up
            [0, -1],  # down
            [-1, 0],  # left
            [1, 0],   # right
        ],
        dtype=np.int32,
    )

    def __init__(
        self,
        grid_size: int = 7,
        num_agents: int = 5,
        max_steps: int = 40,
        vision_range: int = 2,
        collision_penalty: float = 0.5,
        success_bonus: float = 10.0,
        time_penalty: float = 0.1,
        spawn_mode: str = "corners",  # "corners" or "random"
        seed: int | None = None,
    ):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.vision_range = vision_range
        self.collision_penalty = collision_penalty
        self.success_bonus = success_bonus
        self.time_penalty = time_penalty
        self.spawn_mode = spawn_mode
        self.rng = np.random.default_rng(seed)

        # Observation: local grid view (vision_range*2+1)^2 + own position (2) + goal position (2)
        obs_grid_size = (2 * vision_range + 1) ** 2
        self.obs_dim = obs_grid_size + 2 + 2  # grid view + pos + goal
        self.action_dim = len(self.ACTIONS)

        # State
        self.positions = np.zeros((num_agents, 2), dtype=np.int32)
        self.goals = np.zeros((num_agents, 2), dtype=np.int32)
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int32)  # 0=empty, >0=agent_id
        self.t = 0

    def reset(self):
        """Reset environment and return initial observations."""
        self.positions = np.zeros((self.num_agents, 2), dtype=np.int32)
        self.goals = np.zeros((self.num_agents, 2), dtype=np.int32)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.t = 0

        if self.spawn_mode == "corners":
            # Spawn agents at corners, goals at opposite corners
            corners = [
                (0, 0),
                (0, self.grid_size - 1),
                (self.grid_size - 1, 0),
                (self.grid_size - 1, self.grid_size - 1),
            ]
            # Use center if more agents than corners
            if self.num_agents > 4:
                corners.append((self.grid_size // 2, self.grid_size // 2))

            for i in range(self.num_agents):
                corner_idx = i % len(corners)
                self.positions[i] = corners[corner_idx]
                # Goal is opposite corner
                opp_corner = corners[(corner_idx + 2) % len(corners)]
                self.goals[i] = opp_corner
        else:
            # Random spawn
            for i in range(self.num_agents):
                # Spawn at edges
                side = self.rng.integers(4)
                if side == 0:  # top
                    self.positions[i] = [self.rng.integers(self.grid_size), 0]
                elif side == 1:  # bottom
                    self.positions[i] = [self.rng.integers(self.grid_size), self.grid_size - 1]
                elif side == 2:  # left
                    self.positions[i] = [0, self.rng.integers(self.grid_size)]
                else:  # right
                    self.positions[i] = [self.grid_size - 1, self.rng.integers(self.grid_size)]

                # Goal on opposite side
                if side == 0:  # goal at bottom
                    self.goals[i] = [self.rng.integers(self.grid_size), self.grid_size - 1]
                elif side == 1:  # goal at top
                    self.goals[i] = [self.rng.integers(self.grid_size), 0]
                elif side == 2:  # goal at right
                    self.goals[i] = [self.grid_size - 1, self.rng.integers(self.grid_size)]
                else:  # goal at left
                    self.goals[i] = [0, self.rng.integers(self.grid_size)]

        # Update grid
        self._update_grid()
        return self._get_obs()

    def step(self, actions: np.ndarray):
        """Execute one step in the environment."""
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (self.num_agents,)

        # Clear grid
        self.grid.fill(0)

        # Move agents (with collision detection)
        new_positions = np.copy(self.positions)
        for i in range(self.num_agents):
            action = self.ACTIONS[actions[i]]
            new_pos = self.positions[i] + action
            # Clip to grid bounds
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            new_positions[i] = new_pos

        # Resolve collisions: if multiple agents want same cell, none move
        collision_map = {}
        for i in range(self.num_agents):
            pos_key = tuple(new_positions[i])
            if pos_key not in collision_map:
                collision_map[pos_key] = []
            collision_map[pos_key].append(i)

        # Only agents with unique target positions can move
        for pos_key, agent_ids in collision_map.items():
            if len(agent_ids) == 1:
                self.positions[agent_ids[0]] = new_positions[agent_ids[0]]
            # else: collision, agents stay in place

        self._update_grid()
        self.t += 1

        reward, done, info = self._compute_reward_done()
        obs = self._get_obs()
        return obs, reward, done, info

    def _update_grid(self):
        """Update grid state with current agent positions."""
        self.grid.fill(0)
        for i in range(self.num_agents):
            x, y = self.positions[i]
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.grid[y, x] = i + 1  # 1-indexed for agent ID

    def _compute_reward_done(self):
        """Compute reward and done flag."""
        # Check collisions (multiple agents in same cell)
        collision_count = 0
        for i in range(self.num_agents):
            x, y = self.positions[i]
            if self.grid[y, x] > 1:  # More than one agent (value > 1 means collision)
                collision_count += 1

        collision_penalty = self.collision_penalty * collision_count

        # Check if all agents reached goals
        at_goals = np.all(self.positions == self.goals, axis=1)
        all_at_goals = np.all(at_goals)

        reward = -self.time_penalty - collision_penalty
        if all_at_goals:
            reward += self.success_bonus

        done = all_at_goals or self.t >= self.max_steps

        info = {
            "collisions": collision_count,
            "at_goals": int(np.sum(at_goals)),
            "success": all_at_goals,
        }
        return reward, done, info

    def _get_obs(self):
        """Get observations for all agents."""
        obs = []
        for i in range(self.num_agents):
            # Local grid view around agent
            x, y = self.positions[i]
            grid_view = np.zeros((2 * self.vision_range + 1, 2 * self.vision_range + 1), dtype=np.float32)

            for dy in range(-self.vision_range, self.vision_range + 1):
                for dx in range(-self.vision_range, self.vision_range + 1):
                    gx, gy = x + dx, y + dy
                    if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                        # Encode: 0=empty, 1=other agent, 2=goal
                        if self.grid[gy, gx] > 0:
                            if self.grid[gy, gx] == i + 1:
                                grid_view[dy + self.vision_range, dx + self.vision_range] = 0.0  # self
                            else:
                                grid_view[dy + self.vision_range, dx + self.vision_range] = 1.0  # other agent
                        elif gx == self.goals[i][0] and gy == self.goals[i][1]:
                            grid_view[dy + self.vision_range, dx + self.vision_range] = 2.0  # goal
                        else:
                            grid_view[dy + self.vision_range, dx + self.vision_range] = 0.0  # empty
                    # else: out of bounds, stays 0

            # Flatten grid view
            grid_flat = grid_view.flatten()

            # Normalize positions to [0, 1]
            pos_norm = self.positions[i].astype(np.float32) / self.grid_size
            goal_norm = self.goals[i].astype(np.float32) / self.grid_size

            obs_i = np.concatenate([grid_flat, pos_norm, goal_norm], axis=0)
            obs.append(obs_i)

        return np.stack(obs, axis=0)

    def sample_random_action(self):
        """Sample random actions for all agents."""
        return self.rng.integers(0, self.action_dim, size=(self.num_agents,), dtype=np.int64)
