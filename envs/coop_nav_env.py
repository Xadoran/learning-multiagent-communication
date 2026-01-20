import numpy as np


class CooperativeNavEnv:
    """
    Simple cooperative navigation / rendezvous environment.
    - J agents move in a 2D square [-1, 1]^2 with discrete moves.
    - Each agent observes: its position (2), its goal (2), and relative
      positions of other agents (fixed size (J-1)*2), masked to 0 if outside vision.
    - Shared reward each step: negative mean distance to goals minus collision penalty,
      with success bonus if all agents are within goal_eps.
    """

    ACTIONS = np.array(
        [
            [0.0, 0.0],   # stay
            [0.0, 1.0],   # up
            [0.0, -1.0],  # down
            [-1.0, 0.0],  # left
            [1.0, 0.0],   # right
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        num_agents: int = 3,
        step_size: float = 0.25,
        max_steps: int = 40,
        vision_radius: float = 0.4,
        goal_eps: float = 0.2,
        collision_radius: float = 0.1,
        collision_penalty: float = 1.0,
        success_bonus: float = 12.0,
        shared_goal: bool = True,
        time_penalty: float = 0.0,
        spawn_span: float = 0.4,
        progress_scale: float = 2.0,
        seed: int | None = None,
    ):
        self.num_agents = num_agents
        self.step_size = step_size
        self.max_steps = max_steps
        self.vision_radius = vision_radius
        self.goal_eps = goal_eps
        self.collision_radius = collision_radius
        self.collision_penalty = collision_penalty
        self.success_bonus = success_bonus
        self.shared_goal = shared_goal
        self.time_penalty = time_penalty
        self.spawn_span = spawn_span
        self.progress_scale = progress_scale
        self.rng = np.random.default_rng(seed)

        # Observation: own pos (2) + own goal (2) + relative positions of others ((J-1)*2)
        self.obs_dim = 2 + 2 + 2 * (self.num_agents - 1)
        self.action_dim = len(self.ACTIONS)

        self.positions = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.goals = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.t = 0

    def reset(self):
        self.positions = self.rng.uniform(
            -self.spawn_span, self.spawn_span, size=(self.num_agents, 2)
        ).astype(
            np.float32
        )
        if self.shared_goal:
            g = self.rng.uniform(-0.5, 0.5, size=(1, 2)).astype(np.float32)
            self.goals = np.repeat(g, self.num_agents, axis=0)
        else:
            self.goals = self.rng.uniform(
                -self.spawn_span, self.spawn_span, size=(self.num_agents, 2)
            ).astype(
                np.float32
            )
        self.t = 0
        # Track mean distance for shaping
        dists = np.linalg.norm(self.positions - self.goals, axis=1)
        self.last_mean_dist = float(np.mean(dists))
        return self._get_obs()

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (self.num_agents,)

        # Move agents
        deltas = self.ACTIONS[actions] * self.step_size
        self.positions = np.clip(self.positions + deltas, -1.0, 1.0)
        self.t += 1

        reward, done, info = self._compute_reward_done()
        obs = self._get_obs()
        return obs, reward, done, info

    def _compute_reward_done(self):
        dists = np.linalg.norm(self.positions - self.goals, axis=1)
        mean_dist = float(np.mean(dists))

        # Collision penalty if agents are too close
        collision_penalty = 0.0
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if np.linalg.norm(self.positions[i] - self.positions[j]) < self.collision_radius:
                    collision_penalty += self.collision_penalty

        progress = self.last_mean_dist - mean_dist
        reward = self.progress_scale * progress - collision_penalty - self.time_penalty
        success = bool(np.all(dists < self.goal_eps))
        if success:
            reward += self.success_bonus
        self.last_mean_dist = mean_dist

        done = success or self.t >= self.max_steps
        info = {
            "mean_dist": mean_dist,
            "success": success,
            "collisions": collision_penalty > 0.0,
        }
        return reward, done, info

    def _get_obs(self):
        obs = []
        for i in range(self.num_agents):
            rels = []
            for j in range(self.num_agents):
                if i == j:
                    continue
                rel = self.positions[j] - self.positions[i]
                if np.linalg.norm(rel) <= self.vision_radius:
                    rels.append(rel)
                else:
                    rels.append(np.zeros_like(rel))
            flat_rels = np.concatenate(rels, axis=0)
            obs_i = np.concatenate([self.positions[i], self.goals[i], flat_rels], axis=0)
            obs.append(obs_i)
        return np.stack(obs, axis=0)

    def sample_random_action(self):
        return self.rng.integers(0, self.action_dim, size=(self.num_agents,), dtype=np.int64)
