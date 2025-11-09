# quantile_q_learning.py
import numpy as np
from collections import defaultdict
import csv, os, time

class QuantileTable:
    """
    Tabular QR-Q: stores K quantiles per (state, action).
    State = current node index (int). Action = next node index (int).
    """
    def __init__(self, num_states, num_actions, K=51, tau=None, init=0.0):
        self.num_states = num_states
        self.num_actions = num_actions
        self.K = K
        self.tau = np.linspace(1/(2*K), 1 - 1/(2*K), K) if tau is None else np.asarray(tau)
        self.theta = defaultdict(lambda: np.zeros((num_actions, K)) + init)  # theta[s][a, k]

    def mean(self, s):
        # returns mean value per action (K-mean over quantiles)
        qsa = self.theta[s]  # [A,K]
        return qsa.mean(axis=1)  # [A]

    def quantile(self, s, a, q):
        # nearest quantile index
        idx = int(np.clip(round(q * (self.K - 1)), 0, self.K - 1))
        return self.theta[s][a, idx]

    def greedy_action(self, s, mask=None):
        # choose action with minimal mean (delay), respecting mask (True = allowed)
        means = self.mean(s)
        if mask is not None:
            # mask illegal actions with +inf
            means = np.where(mask, means, np.inf)
        return int(np.argmin(means))

def huber(u, kappa=1.0):
    absu = np.abs(u)
    return np.where(absu <= kappa, 0.5 * u**2, kappa * (absu - 0.5 * kappa))

def quantile_td_update(qtab: QuantileTable, s, a, r, s_next, a_next, gamma=1.0, lr=1e-2, kappa=1.0):
    """
    QR Q-learning TD update with fixed taus (Dabney et al.).
    """
    theta_sa = qtab.theta[s][a]                # [K]
    target = r + gamma * qtab.theta[s_next][a_next]  # [K]
    # Compute pairwise TD errors for all quantiles (broadcast [K,K])
    td = target[None, :] - theta_sa[:, None]   # [K,K]
    loss_grad = (qtab.tau[:, None] - (td < 0).astype(float)) * huber(td, kappa=kappa) / kappa
    # gradient step: move theta_sa toward target according to quantile loss
    grad = loss_grad.mean(axis=1)              # [K]
    qtab.theta[s][a] += lr * grad

def epsilon_greedy_safe_action(qtab: QuantileTable, s, feasible_mask, epsilon, omega, deadline_remaining):
    """
    DDRL-style gate: only choose among actions whose (1-omega)-quantile <= remaining deadline.
    Within safe set, act epsilon-greedily using the minimal mean.
    """
    num_actions = feasible_mask.size
    safe = np.zeros(num_actions, dtype=bool)
    for a in range(num_actions):
        if feasible_mask[a]:
            q_high = qtab.quantile(s, a, 1 - omega)
            safe[a] = (q_high <= deadline_remaining)

    # fallback: if nothing is 'safe', use feasible set to avoid deadlocks
    candidate_mask = safe if safe.any() else feasible_mask

    if np.random.rand() < epsilon:
        choices = np.where(candidate_mask)[0]
        return int(np.random.choice(choices))
    # greedy = minimal mean delay
    means = qtab.mean(s)
    means = np.where(candidate_mask, means, np.inf)
    return int(np.argmin(means))

def run_quantile_learning(make_time_sampler,
                          env, G, Final_deadline, num_episodes,
                          omega=0.05, epsilon0=0.2, gamma=1.0, lr=5e-3, K=51):
    """
    make_time_sampler(s, a) -> time_traversed (samples distribution per experiment).
    Writes the same CSV artifacts your TD pipeline expects.
    """
    qtab = QuantileTable(num_states=G.number_of_nodes(),
                         num_actions=G.number_of_nodes(),
                         K=K)

    eps_time = np.empty(num_episodes + 1)
    filename = os.environ['resultsfile']

    for i_episode in range(1, num_episodes + 1):
        start = time.time()
        state = env.reset(Final_deadline)
        total_time = 0.0
        # decaying epsilon like dynamic script
        epsilon = min(1.0, np.exp(-(i_episode / (num_episodes / 10.0))) ) * epsilon0 + 0.01

        for t in range(100):
            deadline = env.get_deadline()

            # build feasible action mask via static wct <= deadline (your current safety gate)
            feasible = np.zeros(G.number_of_nodes(), dtype=bool)
            for j in G[int(state)]:
                feasible[j] = (G[int(state)][j]['wct'] <= deadline)

            # pick action using learned distributional safety (1-omega quantile)
            action = epsilon_greedy_safe_action(qtab, state, feasible, epsilon, omega, deadline)

            # sample time_traversed according to the experiment’s distribution
            time_traversed = make_time_sampler(state, action)

            # env transition (per-hop reward is -time_traversed now)
            next_state, reward, done, _ = env.step(action, time_traversed)
            total_time += time_traversed

            # next action for on-policy QR-SARSA flavor (more stable here)
            # choose greedy next action for target bootstrap (works fine)
            feasible_next = np.zeros(G.number_of_nodes(), dtype=bool)
            for j in G[int(next_state)]:
                feasible_next[j] = True
            if feasible_next.any():
                a_next = qtab.greedy_action(next_state, mask=feasible_next)
            else:
                a_next = action

            quantile_td_update(qtab, state, action, -reward, next_state, a_next, gamma=gamma, lr=lr)

            state = next_state
            if done:
                eps_time[i_episode] = time.time() - start
                break

        # ==== write artifacts like mc_prediction does ====
        # Q_values.csv — we’ll log the mean of quantiles so plots keep working
        with open(filename + "Q_values.csv", "a+", newline="") as f:
            w = csv.writer(f)
            for s in range(G.number_of_nodes()):
                means = qtab.mean(s)
                row = [i_episode, s] + [means[a] for a in range(G.number_of_nodes())]
                w.writerow(row)

        # best_path.csv — follow greedy means from source to sink
        with open(filename + "best_path.csv", "a+", newline="") as f:
            w = csv.writer(f)
            row = [i_episode, G.nodes[0]['name']]
            s = 0
            visited = set()
            for _ in range(G.number_of_nodes()):
                # mask: successors only
                mask = np.zeros(G.number_of_nodes(), dtype=bool)
                for j in G[int(s)]:
                    mask[j] = True
                if not mask.any():
                    break
                a = qtab.greedy_action(s, mask=mask)
                row.append(G.nodes[int(a)]['name'])
                s = a
                if s == G.number_of_nodes() - 1 or s in visited:
                    break
                visited.add(s)
            w.writerow(row)

        # chosen_path.csv is trickier without storing the episode; optional
        # tx_times.csv
        with open(filename + "tx_times.csv", "a+", newline="") as f:
            csv.writer(f).writerow([i_episode, total_time])

        np.savetxt(filename + "comp_times.csv", eps_time, delimiter=",")
    return qtab
