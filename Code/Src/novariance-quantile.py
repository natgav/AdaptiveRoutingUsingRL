# novariance-quantile.py
import os
from environ import RountingEnv
from network_generator import prsnt
from quantile_q_learning import run_quantile_learning

deadline = int(os.environ['deadline'])
num_episodes = int(os.environ['num_episodes'])
omega = float(os.environ.get('omega', '0.05'))

print(f"[QuantileQ] NoVariance: deadline={deadline}, episodes={num_episodes}, omega={omega}")

G = prsnt()
env = RountingEnv(G, deadline)

def make_time_sampler(state, action):
    # No variance: use the deterministic mean transmission time
    return G[int(state)][int(action)]["tx"]

_ = run_quantile_learning(
    make_time_sampler,
    env, G, deadline, num_episodes,
    omega=omega,        #kept for completeness; trainer uses lr
    epsilon0=0.2,
    gamma=1.0,
    lr=2e-3,
    K=31,
    tau_target=0.01
)

