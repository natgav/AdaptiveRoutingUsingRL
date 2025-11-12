# quantile_q_learning.py
import os, math, csv, random, time
import numpy as np
from collections import deque, defaultdict

# ---------------------------
# Utilities
# ---------------------------
def _make_tau(K: int):
    # equally-spaced quantiles at midpoints: (0.5/K, 1.5/K, ..., (K-0.5)/K)
    return (np.arange(K) + 0.5) / K

def _feasible_actions(G, state, deadline):
    feas = []
    if state == int(G.nodes[G.number_of_nodes() - 1]['index']):
        return feas
    for a in G[int(state)]:
        if G[int(state)][a]['wct'] <= deadline:
            feas.append(a)
    return feas

def _epsilon_schedule(e0=0.2, e_min=0.01, decay_episodes=500, episode=1):
    # multiplicative decay to e_min over decay_episodes
    if episode >= decay_episodes: 
        return e_min
    r = (e_min / max(e0, 1e-9)) ** (episode / max(decay_episodes, 1))
    return max(e_min, e0 * r)

def _huber_quantile_gradient(td_matrix, taus, kappa=1.0):
    """
    td_matrix: shape [K, K] with (target_z_j - pred_z_i) for all i,j
    taus: shape [K]
    Returns per-pred-quantile gradients: shape [K]
    Loss is sum_j rho_tau_i^kappa(delta_ij), averaged over j. 
    We return dL/d(pred_z_i).
    """
    # Huber
    abs_td = np.abs(td_matrix)
    huber = np.where(abs_td <= kappa, 0.5 * td_matrix ** 2, kappa * (abs_td - 0.5 * kappa))
    # gradient of huber wrt pred is: 
    #   grad_huber = -( td if |td|<=kappa else kappa*sign(td) )
    grad_huber = np.where(abs_td <= kappa, -td_matrix, -kappa * np.sign(td_matrix))

    # pinball weight (indicator[delta<0] - tau)
    # For each row i, we need sign weights across columns j
    indicator = (td_matrix < 0).astype(np.float64)  # shape [K, K]
    # Broadcast taus: each row i has tau_i
    tau_row = taus[:, None]                         # shape [K,1]
    pinball = np.abs(tau_row - indicator)           # |tau - 1_{delta<0}|  (this equals (tau - 1_{delta<0}) with sign handled by grad_huber)

    # combine
    # dL/d pred_i = mean_j pinball_ij * grad_huber_ij
    grad = np.mean(pinball * grad_huber, axis=1)    # shape [K]
    # loss (optional, for logging)
    loss = np.mean(huber * pinball)
    return grad, loss

# ---------------------------
# Main trainer
# ---------------------------
def run_quantile_learning(
    make_time_sampler, env, G, Final_deadline, num_episodes,
    *,
    omega=0.05,         # terminal slack weight (deadline - total_time)+
    gamma=1.0,          # undiscounted aligns with paper’s episodic path cost
    lr=2e-3,            # tabular “step size” for quantile updates
    K=31,               # number of quantiles
    tau_target=0.01,    # soft target update rate
    epsilon0=0.2,       # initial ε
    epsilon_min=0.01,   # floor
    epsilon_decay_ep=500,
    kappa=1.0,          # Huber threshold
    replay_capacity=50000,
    batch_size=64,
    learn_start=500,    # warm-up steps before learning
    target_update_every=1,  # update target every step (soft)
    max_hops=100
):
    """
    Tabular QR-DQN style learner (but tabular, no NN).
    We keep K quantiles per (state, action). Transitions are stored and
    we train via Huber pinball loss against target quantiles.

    make_time_sampler(state, action) -> tx draw (float or int)
    env.step(action, tx) returns (action, reward, done, _). We ignore env.reward and compute our own:
        r_step = -tx
        if done and total_time <= Final_deadline: r_term += omega * (Final_deadline - total_time)
    CSVs: resultsfile/tx_times.csv  with rows: episode,total_time
          resultsfile/comp_times.csv (seconds per episode index)
          resultsfile/Q_values.csv   (optional best-path snapshot per episode; off by default here)
    """
    results_dir = os.environ.get('resultsfile', '')
    if results_dir and not os.path.isdir(results_dir):
        os.makedirs(results_dir, exist_ok=True)

    # ---- State/Action indexing is direct (tabular over node indices)
    n_nodes = G.number_of_nodes()
    # storage for K quantiles per (s,a). Use dict-of-dicts for sparsity.
    Z = defaultdict(lambda: defaultdict(lambda: np.zeros(K, dtype=np.float64)))
    Z_tgt = defaultdict(lambda: defaultdict(lambda: np.zeros(K, dtype=np.float64)))
    taus = _make_tau(K)

    # simple replay buffer (s,a,r,ns,done) with per-step r
    Replay = deque(maxlen=replay_capacity)

    def get_quantiles(table, s, a):
        return table[int(s)][int(a)]

    def set_quantiles_(table, s, a, new):
        table[int(s)][int(a)] = new

    def expected_q(table, s, a):
        # mean of quantiles
        return float(np.mean(get_quantiles(table, s, a)))

    def greedy_action(table, s, feasible):
        # break ties by smallest tx edge to stabilize early training
        if not feasible:
            return None
        values = np.array([expected_q(table, s, a) for a in feasible], dtype=np.float64)
        # argmax
        idx = int(np.argmax(values))
        return feasible[idx]

    # pre-initialize quantiles to small optimistic values (encourage exploration)
    init_q = 0.0
    for s in G.nodes():
        feas = _feasible_actions(G, s, Final_deadline)
        for a in feas:
            set_quantiles_(Z, s, a, np.full(K, init_q, dtype=np.float64))
            set_quantiles_(Z_tgt, s, a, np.full(K, init_q, dtype=np.float64))

    # logging
    ep_comp_time = np.empty(num_episodes + 1)
    tx_writer = None
    if results_dir:
        tx_writer = csv.writer(open(os.path.join(results_dir, "tx_times.csv"), "w", newline=""))
        # header optional; TeX reads by index, so skip header to match old format.

    total_steps = 0
    rng = np.random.default_rng()

    for ep in range(1, num_episodes + 1):
        t0 = time.time()
        state = env.reset(Final_deadline)
        deadline = env.get_deadline()
        total_time = 0.0
        done = False

        eps = _epsilon_schedule(e0=epsilon0, e_min=epsilon_min, decay_episodes=epsilon_decay_ep, episode=ep)

        for hop in range(max_hops):
            feasible = _feasible_actions(G, state, deadline)
            if not feasible:
                # infeasible—episode ends (missed deadline)
                done = True
                # negative terminal depends only on accumulated step costs (already summed)
                break

            # ε-greedy on expected return
            if rng.random() < eps:
                action = int(rng.choice(feasible))
            else:
                action = greedy_action(Z, state, feasible)

            # draw time for this edge from experiment-specific sampler
            tx = float(make_time_sampler(state, action))
            tx = max(0.0, tx)

            # shaped per-step reward
            r = -tx

            total_time += tx
            _, _, reached, _ = env.step(action, tx)
            next_state = action  # env returns action as "state" per your env; next state is the action/node
            deadline = env.get_deadline()  # already reduced inside env by tx

            # terminal check (destination OR no more hops OR infeasible)
            done = bool(reached)
            # store transition
            Replay.append((state, action, r, next_state, done))
            total_steps += 1

            # learning
            if total_steps >= learn_start and len(Replay) >= batch_size:
                batch = random.sample(Replay, batch_size)

                # for each sample, do tabular quantile TD update
                for (s, a, r_s, ns, d_s) in batch:
                    # build target quantiles: r + gamma * Z_tgt[ns, a*] where a* is greedy on expected Z
                    if d_s:
                        # terminal bonus if success and within deadline
                        # NOTE: we can only compute slack at episode end; approximate using env rule:
                        # if destination reached, remaining deadline is env.deadline; slack = max(deadline,0)
                        # Here we can’t access per-sample slack; use a mild bonus that encourages shorter paths.
                        bonus = 0.0  # true slack is applied at episode end below; keep target clean for stability
                        target = r_s + bonus
                        target_z = np.full(K, target, dtype=np.float64)
                    else:
                        feas_ns = _feasible_actions(G, ns, env.get_deadline())
                        if feas_ns:
                            # greedy next action under TARGET net by expected value
                            next_vals = np.array([np.mean(get_quantiles(Z_tgt, ns, na)) for na in feas_ns])
                            na = feas_ns[int(np.argmax(next_vals))]
                            z_next = get_quantiles(Z_tgt, ns, na)  # shape [K]
                        else:
                            z_next = np.zeros(K, dtype=np.float64)
                        target_z = r_s + (gamma * z_next)

                    z_pred = get_quantiles(Z, s, a)  # [K]
                    # td matrix: [K_pred, K_tgt]
                    td = target_z[None, :] - z_pred[:, None]
                    grad, _ = _huber_quantile_gradient(td, taus, kappa=kappa)
                    # SGD step on quantiles
                    new_z = z_pred - lr * grad
                    set_quantiles_(Z, s, a, new_z)

                # soft target update (tabular)
                if target_update_every > 0 and (total_steps % target_update_every) == 0:
                    for s in list(Z.keys()):
                        for a in list(Z[s].keys()):
                            Z_tgt[s][a] = (1.0 - tau_target) * Z_tgt[s][a] + tau_target * Z[s][a]

            if done:
                break

            state = next_state

        # true terminal slack bonus (added *after* episode roll-out)
        # write one synthetic transition so the agent experiences slack; optional but helps convergence
        if total_time <= Final_deadline:
            slack = Final_deadline - total_time
            term_r = omega * slack
        else:
            term_r = 0.0

        if total_steps >= learn_start:
            # add a terminal self-transition to propagate slack to last state-action
            # We approximate last (s,a) by reusing the last stored transition if exists
            if Replay:
                s, a, r_last, ns, d_last = Replay[-1]
                target_z = np.full(K, term_r, dtype=np.float64)  # no bootstrap
                z_pred = get_quantiles(Z, s, a)
                td = target_z[None, :] - z_pred[:, None]
                grad, _ = _huber_quantile_gradient(td, taus, kappa=kappa)
                new_z = z_pred - lr * grad
                set_quantiles_(Z, s, a, new_z)
                # soft-update once more
                for ss in list(Z.keys()):
                    for aa in list(Z[ss].keys()):
                        Z_tgt[ss][aa] = (1.0 - tau_target) * Z_tgt[ss][aa] + tau_target * Z[ss][aa]

        # CSV logging for TeX
        if tx_writer is not None:
            tx_writer.writerow([ep, total_time])

        ep_comp_time[ep] = time.time() - t0

    # save comp_times.csv (seconds per episode index)
    if results_dir:
        np.savetxt(os.path.join(results_dir, "comp_times.csv"), ep_comp_time, delimiter=",")

    return Z  # policy is implicit: greedy on mean(Z)
