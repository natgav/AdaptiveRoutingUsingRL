# meandelay-uniform-wc-quantile.py
import os, numpy as np
from environ import RountingEnv
from network_generator import prsnt
from quantile_q_learning import run_quantile_learning

deadline = int(os.environ['deadline'])
num_episodes = int(os.environ['num_episodes'])
omega = float(os.environ.get('omega', '0.05'))

print(f"[QuantileQ] Uniform[0,wc]: deadline={deadline}, episodes={num_episodes}, omega={omega}")
G = prsnt()
env = RountingEnv(G, deadline)

def make_time_sampler(state, action):
    wc = G[int(state)][int(action)]["wc"]
    t = int(np.random.uniform(0, wc, 1))
    return max(0, min(t, wc))

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
