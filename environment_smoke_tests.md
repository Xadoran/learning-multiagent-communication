# Environment smoke tests

These are quick “sanity checks” that each environment can be imported, reset, and stepped once with a random action using the project virtualenv (`.venv/bin/python`).

---

## 1) Cooperative Navigation (`--env nav`)

### Command

```bash
.venv/bin/python -c "from envs.coop_nav_env import CooperativeNavEnv; env=CooperativeNavEnv(num_agents=3, seed=0); obs=env.reset(); print('obs',obs.shape,'obs_dim',env.obs_dim,'act_dim',env.action_dim); a=env.sample_random_action(); obs,r,d,info=env.step(a); print('action',a); print('reward',r,'done',d,'info',info)"
```

**What this does:** Creates the navigation environment with 3 agents, resets it, takes one random step, and prints observation shape, action, reward, and info.

### Output

```text
obs (3, 8) obs_dim 8 act_dim 5
action [3 2 2]
reward -0.7654660940170288 done False info {'mean_dist': 0.5588826537132263, 'success': False, 'collisions': False}
```

### Interpretation

- **The observation shape `(3, 8)`** means 3 agents, each seeing an 8-number observation vector.
- **Reward is negative** because the task penalizes distance-to-goal each step; early random moves usually don’t reach the goal.
- **`done False` is expected** after one step because navigation episodes usually run for many steps until success or time limit.

---

## 2) Lever Game (`--env lever`)

### Command

```bash
.venv/bin/python -c "from envs.lever_game_env import LeverGameEnv; env=LeverGameEnv(pool_size=50, active_agents=5, seed=0); obs=env.reset(); print('obs',obs.shape,'obs_dim',env.obs_dim,'act_dim',env.action_dim); a=env.sample_random_action(); obs,r,d,info=env.step(a); print('action',a); print('reward',r,'done',d,'info',info)"
```

**What this does:** Creates the lever environment with 5 active agents sampled from an ID pool of 50, resets it, takes one random joint action, and prints shapes and reward.

### Output

```text
obs (5, 50) obs_dim 50 act_dim 5
action [4 3 4 2 3]
reward 0.6 done True info {'distinct': 3, 'success': False}
```

### Interpretation

- **The observation shape `(5, 50)`** means each of the 5 agents sees a 50-length one-hot ID vector.
- **`done True` is expected** because lever is a one-step game: choose levers once, score once, end immediately.
- **Reward `0.6` matches the rule**: 3 distinct levers out of 5 → \(3/5 = 0.6\).

---

## 3) Traffic Junction (`--env traffic`)

### Command

```bash
.venv/bin/python -c "from envs.traffic_junction_env import TrafficJunctionEnv; env=TrafficJunctionEnv(grid_size=7,num_agents=5,seed=0); obs=env.reset(); print('obs',obs.shape,'obs_dim',env.obs_dim,'act_dim',env.action_dim); a=env.sample_random_action(); obs,r,d,info=env.step(a); print('action',a); print('reward',r,'done',d,'info',info)"
```

**What this does:** Creates the traffic grid environment (7×7) with 5 agents, resets it, takes one random step, and prints the reward and collision info.

### Output

```text
obs (5, 29) obs_dim 29 act_dim 5
action [4 3 2 1 1]
reward -2.1 done False info {'collisions': 4, 'at_goals': 0, 'success': False}
```

### Interpretation

- **The observation shape `(5, 29)`** means 5 agents, each seeing a 29-number local view + position/goal features.
- **A strong negative reward is plausible** because random moves can create collisions, which are penalized.
- **`success False` and `at_goals: 0`** are normal after a single random step.

---

## 4) Predator–Prey (`--env prey`)

### Command

```bash
.venv/bin/python -c "from envs.predator_prey_env import PredatorPreyEnv; env=PredatorPreyEnv(grid_size=7,num_predators=3,num_prey=1,seed=0); obs=env.reset(); print('obs',obs.shape,'obs_dim',env.obs_dim,'act_dim',env.action_dim); a=env.sample_random_action(); obs,r,d,info=env.step(a); print('action',a); print('reward',r,'done',d,'info',info)"
```

**What this does:** Creates predator–prey on a 7×7 grid with 3 predators and 1 prey, resets it, takes one random step, and prints reward and catch status.

### Output

```text
obs (3, 27) obs_dim 27 act_dim 5
action [0 0 0]
reward -0.1 done False info {'caught': 0, 'success': False, 'collisions': 0}
```

### Interpretation

- **The observation shape `(3, 27)`** means 3 predators, each seeing a 27-number local grid window + its position.
- **Reward `-0.1` is expected** because there is typically a small time penalty each step until the prey is caught.
- **`caught: 0` after one step is normal**—catching usually requires coordinated movement over multiple steps.
