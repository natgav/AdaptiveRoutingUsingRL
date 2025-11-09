# meandelay-uniform-quantile.py
import os, numpy as np
from environ import RountingEnv
from network_generator import prsnt
from quantile_q_learning import run_quantile_learning

deadline = int(os.environ['deadline'])
num_episodes = int(os.environ['num_episodes'])
variance = int(os.environ['var'])
omega = float(os.environ.get('omega', '0.05'))

print(f"[QuantileQ] Uniform(mean±var): deadline={deadline}, variance={variance}, episodes={num_episodes}, omega={omega}")
G = prsnt()
env = RountingEnv(G, deadline)

def make_time_sampler(state, action):
    lo = G[int(state)][int(action)]["tx"] - variance
    hi = G[int(state)][int(action)]["tx"] + variance
    wc = G[int(state)][int(action)]["wc"]
    t = int(np.random.uniform(lo, hi, 1))
    return max(0, min(t, wc))

_ = run_quantile_learning(make_time_sampler, env, G, deadline, num_episodes, omega=omega, epsilon0=0.2, gamma=1.0, lr=5e-3, K=51)
