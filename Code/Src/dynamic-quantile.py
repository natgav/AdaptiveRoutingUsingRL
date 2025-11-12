# dynamic-quantile.py
import os, numpy as np, time
from environ import RountingEnv
from network_generator import prsnt
from quantile_q_learning import run_quantile_learning

deadline = int(os.environ['deadline'])
num_episodes = int(os.environ['num_episodes'])
omega = float(os.environ.get('omega', '0.05'))

print(f"[QuantileQ] Dynamic: deadline={deadline}, episodes={num_episodes}, omega={omega}")
G = prsnt()
env = RountingEnv(G, deadline)

_episode = {'i': 0}
def make_time_sampler(state, action):
    # called once per step; use a global-ish episode counter heuristic
    i = _episode['i']
    base = G[int(state)][int(action)]["tx"]
    if i > 40 and action == int(G.nodes[1]['index']):  # same “edge to node 1” offset rule
        return base + 6
    return base

# tiny shim to bump episode counter at the start of each episode
orig_reset = env.reset
def reset_shim(Final_deadline):
    _episode['i'] += 1
    return orig_reset(Final_deadline)
env.reset = reset_shim

#_ = run_quantile_learning(make_time_sampler, env, G, deadline, num_episodes, omega=omega, epsilon0=0.2, gamma=1.0, lr=5e-3, K=51)

_ = run_quantile_learning(
    make_time_sampler,
    env, G, deadline, num_episodes,
    omega=omega,        # kept for completeness; trainer uses lr
    epsilon0=0.2,
    gamma=1.0,
    lr=2e-3,
    K=31,
    tau_target=0.01
)
