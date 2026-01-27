import numpy as np


class TrafficJunctionEnv:
    """
    Traffic Junction environment aligned with the CommNet paper (MazeBase-style).

    - Grid-based 4-way intersection.
    - Cars spawn at the four edges with probability p_arrive, capped at max_cars.
    - Each car is assigned one of three routes (straight, left, right) along a fixed path.
    - Actions: gas (advance one cell along route) or brake (stay).
    - Collision: multiple cars occupying the same cell after a step.
    - Reward per timestep: r(t) = C_t * r_coll + sum_i (tau_i * r_time),
      where tau_i is time since car i spawned.
    - Episode length: max_steps.

    Observation (per agent/car slot):
      - Local 3x3 view (vision_range=1 by default). Each cell encodes:
        [occupancy, multi_car] + ID one-hot (max_cars) + route one-hot (3).
      - Own ID one-hot (max_cars), own position one-hot (grid_size^2), own route one-hot (3).
      Inactive car slots receive zeros.
    """

    # Actions: brake (stay), gas (advance)
    ACTIONS = np.array([0, 1], dtype=np.int64)

    # Route IDs
    ROUTE_STRAIGHT = 0
    ROUTE_RIGHT = 1
    ROUTE_LEFT = 2

    def __init__(
        self,
        grid_size: int = 14,
        max_cars: int = 10,
        max_steps: int = 40,
        p_arrive: float = 0.2,
        vision_range: int = 1,  # 3x3
        r_coll: float = -10.0,
        r_time: float = -0.01,
        seed: int | None = None,
    ):
        self.grid_size = grid_size
        self.max_cars = max_cars
        self.max_steps = max_steps
        self.p_arrive = p_arrive
        self.vision_range = vision_range
        self.r_coll = r_coll
        self.r_time = r_time
        self.rng = np.random.default_rng(seed)

        # Observation dims
        cell_feat = 2 + self.max_cars + 3  # occupancy, multi, ID one-hot, route one-hot
        local_cells = (2 * self.vision_range + 1) ** 2
        self.local_dim = local_cells * cell_feat
        self.obs_dim = self.local_dim + self.max_cars + (self.grid_size ** 2) + 3
        self.action_dim = len(self.ACTIONS)

        # Car state arrays (fixed slots)
        self.active = np.zeros(self.max_cars, dtype=bool)
        self.positions = np.full((self.max_cars, 2), -1, dtype=np.int32)
        self.routes = np.full(self.max_cars, -1, dtype=np.int32)
        self.route_paths = [None] * self.max_cars
        self.route_idx = np.zeros(self.max_cars, dtype=np.int32)
        self.spawn_time = np.zeros(self.max_cars, dtype=np.int32)

        self.t = 0
        self.had_collision = False
        self.cars_exited = 0

        # Precompute paths for each entry direction and route type
        self._paths = self._build_paths()

    def reset(self):
        self.active[:] = False
        self.positions[:] = -1
        self.routes[:] = -1
        self.route_paths = [None] * self.max_cars
        self.route_idx[:] = 0
        self.spawn_time[:] = 0
        self.t = 0
        self.had_collision = False
        self.cars_exited = 0

        # Spawn initial cars (optional small warm start)
        for _ in range(min(4, self.max_cars)):
            self._maybe_spawn(force=True)

        return self._get_obs()

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (self.max_cars,)

        # Spawn new cars (per direction)
        for _ in range(4):
            self._maybe_spawn(force=False)

        # Compute proposed positions
        proposed = np.copy(self.positions)
        for i in range(self.max_cars):
            if not self.active[i]:
                continue
            if actions[i] == 1:  # gas
                path = self.route_paths[i]
                if path is None:
                    continue
                if self.route_idx[i] < len(path) - 1:
                    proposed[i] = path[self.route_idx[i] + 1]

        # Apply moves (simultaneous); collisions allowed (paper says no physical effect)
        self.positions[:] = proposed

        # Count collisions (multiple cars in same cell)
        collision_count = 0
        pos_keys = {}
        for i in range(self.max_cars):
            if not self.active[i]:
                continue
            key = tuple(self.positions[i])
            pos_keys.setdefault(key, []).append(i)
        for key, ids in pos_keys.items():
            if len(ids) > 1:
                collision_count += len(ids)
        if collision_count > 0:
            self.had_collision = True

        # Advance route indices for cars that moved
        for i in range(self.max_cars):
            if not self.active[i]:
                continue
            if actions[i] == 1:
                path = self.route_paths[i]
                if path is None:
                    continue
                if self.route_idx[i] < len(path) - 1:
                    self.route_idx[i] += 1

        # Remove cars that reached the end
        for i in range(self.max_cars):
            if not self.active[i]:
                continue
            path = self.route_paths[i]
            if path is None:
                continue
            if self.route_idx[i] >= len(path) - 1:
                self.cars_exited += 1
                self._despawn(i)

        # Reward
        reward = 0.0
        reward += collision_count * self.r_coll
        for i in range(self.max_cars):
            if not self.active[i]:
                continue
            tau = self.t - self.spawn_time[i]
            reward += tau * self.r_time

        self.t += 1
        done = self.t >= self.max_steps
        info = {
            "collisions": collision_count,
            "active": int(np.sum(self.active)),
            "exited": self.cars_exited,
            "success": not self.had_collision,
        }
        return self._get_obs(), reward, done, info

    # ---- Internal helpers ----

    def _build_paths(self):
        g = self.grid_size
        c = g // 2
        paths = {}
        # Directions: N, S, W, E (incoming)
        # Straight paths
        paths[("N", self.ROUTE_STRAIGHT)] = [(c, y) for y in range(0, g)]
        paths[("S", self.ROUTE_STRAIGHT)] = [(c, y) for y in range(g - 1, -1, -1)]
        paths[("W", self.ROUTE_STRAIGHT)] = [(x, c) for x in range(0, g)]
        paths[("E", self.ROUTE_STRAIGHT)] = [(x, c) for x in range(g - 1, -1, -1)]

        # Right turns
        paths[("N", self.ROUTE_RIGHT)] = [(c, y) for y in range(0, c + 1)] + [(x, c) for x in range(c - 1, -1, -1)]
        paths[("S", self.ROUTE_RIGHT)] = [(c, y) for y in range(g - 1, c - 1, -1)] + [(x, c) for x in range(c + 1, g)]
        paths[("W", self.ROUTE_RIGHT)] = [(x, c) for x in range(0, c + 1)] + [(c, y) for y in range(c - 1, -1, -1)]
        paths[("E", self.ROUTE_RIGHT)] = [(x, c) for x in range(g - 1, c - 1, -1)] + [(c, y) for y in range(c + 1, g)]

        # Left turns
        paths[("N", self.ROUTE_LEFT)] = [(c, y) for y in range(0, c + 1)] + [(x, c) for x in range(c + 1, g)]
        paths[("S", self.ROUTE_LEFT)] = [(c, y) for y in range(g - 1, c - 1, -1)] + [(x, c) for x in range(c - 1, -1, -1)]
        paths[("W", self.ROUTE_LEFT)] = [(x, c) for x in range(0, c + 1)] + [(c, y) for y in range(c + 1, g)]
        paths[("E", self.ROUTE_LEFT)] = [(x, c) for x in range(g - 1, c - 1, -1)] + [(c, y) for y in range(c - 1, -1, -1)]
        return paths

    def _maybe_spawn(self, force: bool = False):
        if np.sum(self.active) >= self.max_cars:
            return
        if not force and self.rng.random() > self.p_arrive:
            return
        # Choose a direction to spawn
        direction = self.rng.choice(["N", "S", "W", "E"])
        route = self.rng.integers(0, 3)
        path = self._paths[(direction, route)]
        spawn_pos = path[0]

        # Find empty slot
        slot = np.where(~self.active)[0][0]
        self.active[slot] = True
        self.positions[slot] = np.array(spawn_pos, dtype=np.int32)
        self.routes[slot] = route
        self.route_paths[slot] = path
        self.route_idx[slot] = 0
        self.spawn_time[slot] = self.t

    def _despawn(self, idx: int):
        self.active[idx] = False
        self.positions[idx] = -1
        self.routes[idx] = -1
        self.route_paths[idx] = None
        self.route_idx[idx] = 0
        self.spawn_time[idx] = 0

    def _get_obs(self):
        obs = []
        g = self.grid_size
        v = self.vision_range
        cell_feat = 2 + self.max_cars + 3

        # Build a map of cell -> list of car ids in that cell
        cell_map = {}
        for i in range(self.max_cars):
            if not self.active[i]:
                continue
            key = tuple(self.positions[i])
            cell_map.setdefault(key, []).append(i)

        for i in range(self.max_cars):
            if not self.active[i]:
                obs.append(np.zeros(self.obs_dim, dtype=np.float32))
                continue

            x, y = self.positions[i]
            local = []
            for dy in range(-v, v + 1):
                for dx in range(-v, v + 1):
                    gx, gy = x + dx, y + dy
                    cell_vec = np.zeros(cell_feat, dtype=np.float32)
                    if 0 <= gx < g and 0 <= gy < g:
                        key = (gx, gy)
                        if key in cell_map:
                            ids = cell_map[key]
                            cell_vec[0] = 1.0  # occupancy
                            if len(ids) > 1:
                                cell_vec[1] = 1.0  # multi-car
                            # Encode first car ID and route
                            cid = ids[0]
                            cell_vec[2 + cid] = 1.0
                            r = self.routes[cid]
                            if r >= 0:
                                cell_vec[2 + self.max_cars + r] = 1.0
                    local.append(cell_vec)

            local_flat = np.concatenate(local, axis=0)

            # Own ID one-hot
            own_id = np.zeros(self.max_cars, dtype=np.float32)
            own_id[i] = 1.0
            # Own position one-hot
            pos_onehot = np.zeros(g * g, dtype=np.float32)
            pos_onehot[y * g + x] = 1.0
            # Own route one-hot
            own_route = np.zeros(3, dtype=np.float32)
            if self.routes[i] >= 0:
                own_route[self.routes[i]] = 1.0

            obs_i = np.concatenate([local_flat, own_id, pos_onehot, own_route], axis=0)
            obs.append(obs_i)

        return np.stack(obs, axis=0)

    def sample_random_action(self):
        return self.rng.integers(0, self.action_dim, size=(self.max_cars,), dtype=np.int64)
