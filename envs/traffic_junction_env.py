import numpy as np


class TrafficJunctionEnv:
    """
    Traffic Junction task (CommNet, NIPS 2016) implemented in a fixed-slot way.

    Paper spec (core points):
    - 14x14 grid, 4-way junction.
    - New cars arrive each time step with probability p_arrive from each direction,
      capped by Nmax cars present.
    - Each car assigned to one of 3 routes (right turn / straight / left turn), and
      has 2 actions: GAS (advance 1 along route) or BRAKE (stay).
    - Collision if two cars overlap -> r_coll = -10 per collision event.
      Collision does not affect simulation otherwise, but the episode is a failure if
      >=1 collision occurred in the 40-step horizon.
    - Time penalty: for each car i present at time t, add tau_i * r_time where
      r_time = -0.01 and tau_i is time since the car arrived.

    Compatibility choices (to fit your current train loop):
    - We keep a fixed number of agent slots = Nmax. "Inactive" slots represent no car.
    - Actions for inactive slots are ignored.
    - Observation is fixed-size per slot; slots that are inactive emit all-zeros.

    Observation (per slot):
    - present flag (1)
    - self ID one-hot (Nmax)
    - self location one-hot (grid_size*grid_size)
    - self route one-hot (3)
    - self tau normalized scalar (1)
    - local 3x3 neighborhood features (9 cells):
        for each cell: neighbor_present(1) + neighbor_id_onehot(Nmax) + neighbor_route_onehot(3)
      (masked to 0 if empty / out of bounds)

    Reward:
      r(t) = C_t * r_coll + sum_{cars i present} tau_i * r_time
      where C_t is the number of pairwise collisions at time t.

    Done:
      - after max_steps=40
      - info['failure'] True iff any collision occurred at any time
    """

    # Actions
    BRAKE = 0
    GAS = 1

    def __init__(
        self,
        grid_size: int = 14,
        nmax: int = 10,
        max_steps: int = 40,
        p_arrive: float = 0.2,
        r_coll: float = -10.0,
        r_time: float = -0.01,
        vision_range: int = 1,  # 3x3 neighborhood
        seed: int | None = None,
    ):
        assert grid_size == 14, "CommNet paper uses 14x14 for Traffic Junction"
        assert vision_range == 1, "CommNet paper uses 3x3 neighborhood (vision_range=1)"

        self.grid_size = grid_size
        self.nmax = nmax
        self.max_steps = max_steps
        self.p_arrive = float(p_arrive)
        self.r_coll = float(r_coll)
        self.r_time = float(r_time)
        self.vision_range = vision_range
        self.rng = np.random.default_rng(seed)

        # Fixed action space: brake/gas
        self.action_dim = 2

        # Obs dims
        self.id_dim = self.nmax
        self.loc_dim = self.grid_size * self.grid_size
        self.route_dim = 3

        # per neighbor cell: present(1) + id_onehot(Nmax) + route_onehot(3)
        self.nei_cell_dim = 1 + self.id_dim + self.route_dim
        self.nei_dim = (
            2 * self.vision_range + 1
        ) ** 2 * self.nei_cell_dim  # 9 * cell_dim

        # per agent:
        # present(1) + id_onehot + loc_onehot + route_onehot + tau_scalar(1) + neighborhood
        self.obs_dim = (
            1 + self.id_dim + self.loc_dim + self.route_dim + 1 + self.nei_dim
        )

        # State arrays for fixed slots
        self.present = np.zeros((self.nmax,), dtype=np.bool_)
        self.pos = np.zeros((self.nmax, 2), dtype=np.int32)  # (x,y)
        self.route_id = np.zeros((self.nmax,), dtype=np.int32)  # 0..2
        self.route_idx = np.zeros(
            (self.nmax,), dtype=np.int32
        )  # index along route path
        self.tau = np.zeros((self.nmax,), dtype=np.int32)  # time since arrival

        self.t = 0
        self.any_collision = False

        # Precompute routes: for each entry dir and route_id -> list of cells (x,y)
        # Entry points are placed around the intersection centered near grid center.
        self._routes = self._build_all_routes()

    def reset(self):
        self.present.fill(False)
        self.pos.fill(0)
        self.route_id.fill(0)
        self.route_idx.fill(0)
        self.tau.fill(0)

        self.t = 0
        self.any_collision = False

        # Optionally spawn a few cars at start (paper depicts ongoing arrivals;
        # starting empty is fine). We'll start empty.
        return self._get_obs()

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int64)
        assert actions.shape == (self.nmax,)

        # 1) Arrivals (up to 4 directions), constrained by Nmax
        self._spawn_arrivals()

        # 2) Apply actions (simultaneous): move GAS cars forward by 1 along their route
        #    BRAKE does nothing. Inactive slots ignored.
        new_pos = self.pos.copy()
        new_route_idx = self.route_idx.copy()

        for i in range(self.nmax):
            if not self.present[i]:
                continue
            a = int(actions[i])
            if a == self.GAS:
                rid = int(self.route_id[i])
                entry = self._infer_entry_from_route(self._route_key_for_slot(i))
                path = self._routes[(entry, rid)]
                nxt = int(self.route_idx[i]) + 1
                if nxt < len(path):
                    new_route_idx[i] = nxt
                    new_pos[i] = path[nxt]
                else:
                    # Already at end: will be removed below
                    pass
            # BRAKE => stay

        # Commit movement
        self.pos[:] = new_pos
        self.route_idx[:] = new_route_idx

        # 3) Remove cars that reached end of their path (exited grid)
        self._remove_exited()

        # 4) Compute collisions at current time step (pairwise overlaps)
        c_t = self._count_pairwise_collisions()
        if c_t > 0:
            self.any_collision = True

        # 5) Time penalties and tau update
        # Reward uses tau at this time step (time since arrival).
        time_pen = 0.0
        for i in range(self.nmax):
            if self.present[i]:
                time_pen += float(self.tau[i]) * self.r_time

        reward = float(c_t) * self.r_coll + float(time_pen)

        # Increase tau for cars still present
        for i in range(self.nmax):
            if self.present[i]:
                self.tau[i] += 1

        self.t += 1
        done = self.t >= self.max_steps

        info = {
            "collisions_t": int(c_t),
            "failure": bool(self.any_collision),
            "cars_present": int(self.present.sum()),
            "success": bool(done and not self.any_collision),
        }

        obs = self._get_obs()
        return obs, reward, done, info

    def sample_random_action(self):
        return self.rng.integers(0, self.action_dim, size=(self.nmax,), dtype=np.int64)

    # -------------------------
    # Internal helpers
    # -------------------------

    def _spawn_arrivals(self):
        # Directions: N,S,W,E (cars enter from edge heading toward junction)
        # We'll attempt arrivals from each direction with p_arrive.
        directions = ["N", "S", "W", "E"]
        for d in directions:
            if self.present.sum() >= self.nmax:
                break
            if self.rng.random() < self.p_arrive:
                slot = self._first_free_slot()
                if slot is None:
                    break
                rid = int(self.rng.integers(0, 3))  # 0..2 route id
                path = self._routes[(d, rid)]
                # spawn at path[0]
                self.present[slot] = True
                self.route_id[slot] = rid
                self.route_idx[slot] = 0
                self.pos[slot] = path[0]
                self.tau[slot] = 0
                # record route-key for this slot (direction is not otherwise stored)
                self._set_route_key_for_slot(slot, d)

    def _first_free_slot(self):
        free = np.where(~self.present)[0]
        return int(free[0]) if free.size > 0 else None

    def _remove_exited(self):
        # A car exits when it reaches the last cell in its path and GAS again would go out.
        # Here we remove if route_idx is already at the final cell.
        for i in range(self.nmax):
            if not self.present[i]:
                continue
            d = self._route_key_for_slot(i)
            rid = int(self.route_id[i])
            path = self._routes[(d, rid)]
            if int(self.route_idx[i]) >= len(path) - 1:
                # remove
                self.present[i] = False
                self.pos[i] = 0
                self.route_idx[i] = 0
                self.route_id[i] = 0
                self.tau[i] = 0
                self._set_route_key_for_slot(i, "N")  # default placeholder

    def _count_pairwise_collisions(self) -> int:
        # Count pairwise collisions at this time step:
        # if k cars occupy same cell -> k choose 2 collisions.
        coords = {}
        for i in range(self.nmax):
            if not self.present[i]:
                continue
            key = (int(self.pos[i][0]), int(self.pos[i][1]))
            coords[key] = coords.get(key, 0) + 1
        c = 0
        for k in coords.values():
            if k >= 2:
                c += k * (k - 1) // 2
        return int(c)

    def _get_obs(self):
        obs = np.zeros((self.nmax, self.obs_dim), dtype=np.float32)

        for i in range(self.nmax):
            if not self.present[i]:
                continue

            feats = []

            # present
            feats.append(np.array([1.0], dtype=np.float32))

            # id one-hot (slot index is the unique ID in this fixed-slot version)
            id_oh = np.zeros((self.id_dim,), dtype=np.float32)
            id_oh[i] = 1.0
            feats.append(id_oh)

            # location one-hot
            loc_oh = np.zeros((self.loc_dim,), dtype=np.float32)
            x, y = int(self.pos[i][0]), int(self.pos[i][1])
            loc_oh[y * self.grid_size + x] = 1.0
            feats.append(loc_oh)

            # route one-hot
            r_oh = np.zeros((self.route_dim,), dtype=np.float32)
            r_oh[int(self.route_id[i])] = 1.0
            feats.append(r_oh)

            # tau normalized (divide by max_steps)
            feats.append(
                np.array([float(self.tau[i]) / float(self.max_steps)], dtype=np.float32)
            )

            # neighborhood (3x3)
            neigh = []
            cx, cy = x, y
            for dy in range(-self.vision_range, self.vision_range + 1):
                for dx in range(-self.vision_range, self.vision_range + 1):
                    gx, gy = cx + dx, cy + dy
                    cell = np.zeros((self.nei_cell_dim,), dtype=np.float32)
                    if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                        # find if another car occupies that cell
                        occ = self._car_at_cell_excluding(gx, gy, exclude=i)
                        if occ is not None:
                            cell[0] = 1.0  # neighbor_present
                            nid = np.zeros((self.id_dim,), dtype=np.float32)
                            nid[occ] = 1.0
                            cell[1 : 1 + self.id_dim] = nid
                            nr = np.zeros((self.route_dim,), dtype=np.float32)
                            nr[int(self.route_id[occ])] = 1.0
                            cell[1 + self.id_dim :] = nr
                    neigh.append(cell)
            feats.append(np.concatenate(neigh, axis=0))

            obs[i] = np.concatenate(feats, axis=0)

        return obs

    def _car_at_cell_excluding(self, x: int, y: int, exclude: int):
        for j in range(self.nmax):
            if j == exclude or not self.present[j]:
                continue
            if int(self.pos[j][0]) == x and int(self.pos[j][1]) == y:
                return j
        return None

    # --- Route construction (simple right-hand lanes) ---

    def _build_all_routes(self):
        # We build 3 route types from each entry direction:
        # 0: right turn, 1: straight, 2: left turn
        # (This matches “3 possible routes” in the paper.) :contentReference[oaicite:2]{index=2}
        routes = {}
        for d in ["N", "S", "W", "E"]:
            for rid in [0, 1, 2]:
                routes[(d, rid)] = self._build_route(d, rid)
        return routes

    def _build_route(self, entry: str, rid: int):
        """
        Returns a list of (x,y) cells describing the route path.
        This is a lightweight implementation: single-lane roads through the center.
        """
        g = self.grid_size
        cx = g // 2  # 7
        cy = g // 2  # 7

        # We'll place lanes slightly offset to keep "right-hand side" behavior consistent.
        # Using two center lines: vertical lane at x=cx-1, horizontal lane at y=cy-1.
        vx = cx - 1
        hy = cy - 1

        path = []

        if entry == "N":
            # spawn at top going down
            x0, y0 = vx, 0
            if rid == 0:  # right turn: N -> W
                # down to junction then left to edge
                path += [(x0, y) for y in range(y0, hy + 1)]
                path += [(x, hy) for x in range(x0 - 1, -1, -1)]
            elif rid == 1:  # straight: N -> S
                path += [(x0, y) for y in range(y0, g)]
            else:  # left turn: N -> E
                path += [(x0, y) for y in range(y0, hy + 1)]
                path += [(x, hy) for x in range(x0 + 1, g)]
        elif entry == "S":
            x0, y0 = cx, g - 1
            if rid == 0:  # right turn: S -> E
                path += [(x0, y) for y in range(y0, hy - 1, -1)]
                path += [(x, hy) for x in range(x0 + 1, g)]
            elif rid == 1:  # straight: S -> N
                path += [(x0, y) for y in range(y0, -1, -1)]
            else:  # left turn: S -> W
                path += [(x0, y) for y in range(y0, hy - 1, -1)]
                path += [(x, hy) for x in range(x0 - 1, -1, -1)]
        elif entry == "W":
            x0, y0 = 0, hy
            if rid == 0:  # right turn: W -> N
                path += [(x, y0) for x in range(x0, vx + 1)]
                path += [(vx, y) for y in range(y0 - 1, -1, -1)]
            elif rid == 1:  # straight: W -> E
                path += [(x, y0) for x in range(x0, g)]
            else:  # left turn: W -> S
                path += [(x, y0) for x in range(x0, vx + 1)]
                path += [(vx, y) for y in range(y0 + 1, g)]
        else:  # entry == "E"
            x0, y0 = g - 1, cy
            if rid == 0:  # right turn: E -> S
                path += [(x, y0) for x in range(x0, vx - 1, -1)]
                path += [(vx, y) for y in range(y0 + 1, g)]
            elif rid == 1:  # straight: E -> W
                path += [(x, y0) for x in range(x0, -1, -1)]
            else:  # left turn: E -> N
                path += [(x, y0) for x in range(x0, vx - 1, -1)]
                path += [(vx, y) for y in range(y0 - 1, -1, -1)]

        # Ensure path stays within bounds and is at least length 2
        path = [(int(x), int(y)) for (x, y) in path if 0 <= x < g and 0 <= y < g]
        if len(path) < 2:
            path = [(vx, 0), (vx, 1)]
        return np.array(path, dtype=np.int32)

    # --- We store entry direction per slot (since route path depends on entry) ---
    # This keeps the env fully self-contained without extra arrays exposed.

    def _init_route_keys(self):
        self._slot_entry = np.array(["N"] * self.nmax, dtype=object)

    def _route_key_for_slot(self, i: int) -> str:
        if not hasattr(self, "_slot_entry"):
            self._init_route_keys()
        return str(self._slot_entry[i])

    def _set_route_key_for_slot(self, i: int, entry: str):
        if not hasattr(self, "_slot_entry"):
            self._init_route_keys()
        self._slot_entry[i] = entry

    def _infer_entry_from_route(self, entry: str) -> str:
        # stored already
        return entry
