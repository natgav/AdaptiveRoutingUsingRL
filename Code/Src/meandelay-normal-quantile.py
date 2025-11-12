# meandelay-normal-quantile.py
import os, numpy as np
import scipy.stats as stats
from environ import RountingEnv
from network_generator import prsnt
from quantile_q_learning import run_quantile_learning

deadline = int(os.environ['deadline'])
num_episodes = int(os.environ['num_episodes'])
variance = int(os.environ['var'])
omega = float(os.environ.get('omega', '0.05'))  # deadline miss tolerance

print(f"[QuantileQ] Normal: deadline={deadline}, variance={variance}, episodes={num_episodes}, omega={omega}")
G = prsnt()
env = RountingEnv(G, deadline)

def make_time_sampler(state, action):
    lower, upper = 0, G[int(state)][int(action)]["wc"]
    mu, sigma = G[int(state)][int(action)]["tx"], variance
    return stats.truncnorm.rvs((lower - mu)/sigma, (upper - mu)/sigma, loc=mu, scale=sigma)

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
