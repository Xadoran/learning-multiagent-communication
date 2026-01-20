import numpy as np


class LeverGameEnv:
    """
    Lever-pulling coordination game (CommNet paper).
    - Pool of N agent IDs; each episode samples M active agents (M <= N).
    - Each active agent observes only its own ID (one-hot over pool size).
    - All active agents choose a lever simultaneously (num_levers default = M).
    - Shared reward = (# distinct levers chosen) / num_levers.
    - Episode ends after one step.
    """

    def __init__(
        self,
        pool_size: int = 500,
        active_agents: int = 5,
        num_levers: int | None = None,
        seed: int | None = None,
    ):
        assert active_agents <= pool_size, "active_agents must be <= pool_size"
        self.pool_size = pool_size
        self.active_agents = active_agents
        self.num_levers = num_levers if num_levers is not None else active_agents
        self.rng = np.random.default_rng(seed)
        # Observation: one-hot agent ID over pool_size
        self.obs_dim = self.pool_size
        self.action_dim = self.num_levers
        self.current_ids = None

    def reset(self):
        # Sample active agent IDs without replacement
        self.current_ids = self.rng.choice(self.pool_size, size=self.active_agents, replace=False)
        obs = np.zeros((self.active_agents, self.pool_size), dtype=np.float32)
        obs[np.arange(self.active_agents), self.current_ids] = 1.0
        return obs

    def optimal_actions(self):
        """
        Deterministic optimal mapping: sort sampled agent IDs and assign levers in order.
        Returns: np.ndarray shape (active_agents,)
        """
        assert self.current_ids is not None, "Call reset() before optimal_actions()"
        order = np.argsort(self.current_ids)
        actions = np.zeros(self.active_agents, dtype=np.int64)
        for rank, idx in enumerate(order):
            actions[idx] = rank % self.num_levers
        return actions

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (self.active_agents,)
        distinct = len(set(actions.tolist()))
        reward = distinct / float(self.num_levers)
        done = True  # single-step episode
        info = {"distinct": distinct}
        obs = self.reset()  # stateless across episodes
        return obs, reward, done, info

    def sample_random_action(self):
        return self.rng.integers(0, self.action_dim, size=(self.active_agents,), dtype=np.int64)
