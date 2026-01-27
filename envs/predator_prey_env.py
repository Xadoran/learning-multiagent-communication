import numpy as np


class PredatorPreyEnv:
    """
    Predator-Prey hunting environment (CommNet paper).
    - Grid-based environment where predators (agents) must coordinate to catch prey.
    - Predators spawn randomly and need to surround/catch prey.
    - Partial observability: agents observe local grid around them.
    - Shared reward: success bonus when prey is caught, minus time penalty.
    - Prey moves randomly or stays still (configurable).
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
        num_predators: int = 3,
        num_prey: int = 1,
        max_steps: int = 40,
        vision_range: int = 2,
        catch_radius: float = 1.0,  # Distance to catch prey (1.0 = adjacent, 1.41 = diagonal)
        success_bonus: float = 10.0,
        time_penalty: float = 0.1,
        prey_move_prob: float = 0.5,  # Probability prey moves each step
        collision_penalty: float = 0.2,
        seed: int | None = None,
    ):
        self.grid_size = grid_size
        self.num_predators = num_predators
        self.num_prey = num_prey
        self.max_steps = max_steps
        self.vision_range = vision_range
        self.catch_radius = catch_radius
        self.success_bonus = success_bonus
        self.time_penalty = time_penalty
        self.prey_move_prob = prey_move_prob
        self.collision_penalty = collision_penalty
        self.rng = np.random.default_rng(seed)

        # Observation: local grid view (vision_range*2+1)^2 + own position (2)
        obs_grid_size = (2 * vision_range + 1) ** 2
        self.obs_dim = obs_grid_size + 2  # grid view + pos
        self.action_dim = len(self.ACTIONS)

        # State
        self.predator_positions = np.zeros((num_predators, 2), dtype=np.int32)
        self.prey_positions = np.zeros((num_prey, 2), dtype=np.int32)
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int32)  # 0=empty, 1=predator, 2=prey
        self.t = 0
        self.last_collision_count = 0

    def reset(self):
        """Reset environment and return initial observations."""
        self.predator_positions = np.zeros((self.num_predators, 2), dtype=np.int32)
        self.prey_positions = np.zeros((self.num_prey, 2), dtype=np.int32)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.t = 0

        # Spawn predators randomly (avoid center initially)
        spawn_positions = []
        for i in range(self.num_predators):
            while True:
                pos = self.rng.integers(0, self.grid_size, size=2)
                # Avoid center region for initial spawn
                center = self.grid_size // 2
                if np.linalg.norm(pos - center) > 1:
                    if tuple(pos) not in spawn_positions:
                        spawn_positions.append(tuple(pos))
                        self.predator_positions[i] = pos
                        break

        # Spawn prey in center or random
        for i in range(self.num_prey):
            if i == 0:
                # First prey in center
                self.prey_positions[i] = [self.grid_size // 2, self.grid_size // 2]
            else:
                # Additional prey randomly
                while True:
                    pos = self.rng.integers(0, self.grid_size, size=2)
                    if tuple(pos) not in [tuple(p) for p in self.prey_positions[:i]]:
                        self.prey_positions[i] = pos
                        break

        self._update_grid()
        self.last_collision_count = 0
        return self._get_obs()

    def step(self, actions: np.ndarray):
        """Execute one step in the environment."""
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (self.num_predators,)

        # Clear grid
        self.grid.fill(0)

        # Move predators
        new_predator_positions = np.copy(self.predator_positions)
        for i in range(self.num_predators):
            action = self.ACTIONS[actions[i]]
            new_pos = self.predator_positions[i] + action
            # Clip to grid bounds
            new_pos = np.clip(new_pos, 0, self.grid_size - 1)
            new_predator_positions[i] = new_pos

        # Resolve predator collisions: if multiple want same cell, none move
        collision_map = {}
        for i in range(self.num_predators):
            pos_key = tuple(new_predator_positions[i])
            if pos_key not in collision_map:
                collision_map[pos_key] = []
            collision_map[pos_key].append(i)

        collision_count = 0
        for pos_key, agent_ids in collision_map.items():
            if len(agent_ids) == 1:
                self.predator_positions[agent_ids[0]] = new_predator_positions[agent_ids[0]]
            else:
                collision_count += len(agent_ids)

        # Move prey (random movement)
        for i in range(self.num_prey):
            if self.rng.random() < self.prey_move_prob:
                # Prey moves randomly
                action = self.ACTIONS[self.rng.integers(len(self.ACTIONS))]
                new_pos = self.prey_positions[i] + action
                new_pos = np.clip(new_pos, 0, self.grid_size - 1)
                # Prey avoids predators if possible
                if not self._is_predator_at(new_pos):
                    self.prey_positions[i] = new_pos

        self._update_grid()
        self.t += 1
        self.last_collision_count = collision_count

        reward, done, info = self._compute_reward_done()
        obs = self._get_obs()
        return obs, reward, done, info

    def _is_predator_at(self, pos):
        """Check if any predator is at given position."""
        return np.any(np.all(self.predator_positions == pos, axis=1))

    def _update_grid(self):
        """Update grid state with current positions."""
        self.grid.fill(0)
        for i in range(self.num_predators):
            x, y = self.predator_positions[i]
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.grid[y, x] = 1  # predator
        for i in range(self.num_prey):
            x, y = self.prey_positions[i]
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.grid[y, x] = 2  # prey

    def _compute_reward_done(self):
        """Compute reward and done flag."""
        # Check if prey are caught
        caught_count = 0
        for i in range(self.num_prey):
            prey_pos = self.prey_positions[i]
            # Count predators within catch radius
            distances = np.linalg.norm(self.predator_positions - prey_pos, axis=1)
            nearby_predators = np.sum(distances <= self.catch_radius)
            if nearby_predators >= 2:  # Need at least 2 predators to catch
                caught_count += 1

        all_caught = caught_count == self.num_prey

        collision_penalty = self.collision_penalty * self.last_collision_count

        reward = -self.time_penalty - collision_penalty
        if all_caught:
            reward += self.success_bonus

        done = all_caught or self.t >= self.max_steps

        info = {
            "caught": caught_count,
            "success": all_caught,
            "collisions": self.last_collision_count,
        }
        return reward, done, info

    def _get_obs(self):
        """Get observations for all predators."""
        obs = []
        for i in range(self.num_predators):
            # Local grid view around predator
            x, y = self.predator_positions[i]
            grid_view = np.zeros((2 * self.vision_range + 1, 2 * self.vision_range + 1), dtype=np.float32)

            for dy in range(-self.vision_range, self.vision_range + 1):
                for dx in range(-self.vision_range, self.vision_range + 1):
                    gx, gy = x + dx, y + dy
                    if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                        # Encode: 0=empty, 1=other predator, 2=prey
                        if self.grid[gy, gx] == 1:
                            # Check if it's self or other predator
                            if gx == x and gy == y:
                                grid_view[dy + self.vision_range, dx + self.vision_range] = 0.0  # self
                            else:
                                grid_view[dy + self.vision_range, dx + self.vision_range] = 1.0  # other predator
                        elif self.grid[gy, gx] == 2:
                            grid_view[dy + self.vision_range, dx + self.vision_range] = 2.0  # prey
                        else:
                            grid_view[dy + self.vision_range, dx + self.vision_range] = 0.0  # empty
                    # else: out of bounds, stays 0

            # Flatten grid view
            grid_flat = grid_view.flatten()

            # Normalize position to [0, 1]
            pos_norm = self.predator_positions[i].astype(np.float32) / self.grid_size

            obs_i = np.concatenate([grid_flat, pos_norm], axis=0)
            obs.append(obs_i)

        return np.stack(obs, axis=0)

    def sample_random_action(self):
        """Sample random actions for all predators."""
        return self.rng.integers(0, self.action_dim, size=(self.num_predators,), dtype=np.int64)
