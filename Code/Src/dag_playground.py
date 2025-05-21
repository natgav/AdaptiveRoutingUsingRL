#!/usr/bin/env python3
"""
dag_playground.py
-----------------
Quick utility to generate and visualise *random* directed‑acyclic graphs (DAGs)
similar to those used in the Adaptive‑Routing RL paper.

• Each edge is annotated with
      tx  – “typical” transmission delay
      wc  – worst‑case delay
      wct – worst‑case delay **plus** the shortest remaining wc‑path
            to the sink (pre‑computed for fast feasibility checks)

Usage
~~~~~
$ python dag_playground.py --nodes 30 --edges 120 --count 3 --seed 42 --plot
"""
import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# --------------------------------------------------------------------------- #
#  Graph generator                                                             #
# --------------------------------------------------------------------------- #
def create_random_dag(
    n_nodes: int,
    n_extra_edges: int,
    *,
    rng: random.Random,
    tx_range=(4, 10),
    wc_range=(10, 30),
) -> nx.DiGraph:
    """
    Build a random DAG with *n_nodes* and approximately
    (n_nodes‑1) + n_extra_edges edges.

    The graph always contains
        0  ........  source  node  “i”
        n_nodes‑1 .. sink / destination node  “t”
    """
    G = nx.DiGraph()

    # --- 1.  add nodes, label 0 … n_nodes‑1 ------------------------------- #
    for idx in range(n_nodes):
        G.add_node(idx, index=str(idx))

    G.nodes[0]["name"] = "i"
    G.nodes[n_nodes - 1]["name"] = "t"

    # --- 2.  helper: random edge attributes -------------------------------- #
    def _rand_tx_wc():
        return (
            rng.randint(*tx_range),
            rng.randint(*wc_range),
        )

    # --- 3.  mandatory start & end edges  ---------------------------------- #
    tx, wc = _rand_tx_wc()
    G.add_edge(0, 1, tx=tx, wc=wc)  # ensure out‑edge from source

    tx, wc = _rand_tx_wc()
    G.add_edge(
        n_nodes - 2, n_nodes - 1, tx=tx, wc=wc
    )  # ensure in‑edge to sink

    # --- 4.  add extra edges without introducing cycles -------------------- #
    edges_to_add = n_extra_edges
    while edges_to_add > 0:
        u = rng.randint(0, n_nodes - 2)
        v = rng.randint(u + 1, n_nodes - 1)  # v > u keeps DAG property
        if G.has_edge(u, v):
            continue  # skip duplicates

        tx, wc = _rand_tx_wc()
        G.add_edge(u, v, tx=tx, wc=wc)
        edges_to_add -= 1

    # --- 5.  ensure every node can reach sink ------------------------------ #
    for node in range(n_nodes - 1):
        if not nx.has_path(G, node, n_nodes - 1):
            # add a direct edge to sink
            tx, wc = _rand_tx_wc()
            G.add_edge(node, n_nodes - 1, tx=tx, wc=wc)

    # --- 6.  pre‑compute “wct” (wc + downstream worst‑case distance) ------- #
    # Compute single‑source (actually single‑*sink*) dijkstra on the reversed
    # graph so that path lengths *to* sink become path lengths *from* sink.
    revG = G.reverse(copy=False)
    wc_dist = nx.single_source_dijkstra_path_length(
        revG, source=n_nodes - 1, weight="wc"
    )

    for u, v, data in G.edges(data=True):
        data["wct"] = data["wc"] + wc_dist.get(v, float("inf"))

    return G


# --------------------------------------------------------------------------- #
#  CLI / visualisation                                                        #
# --------------------------------------------------------------------------- #
def draw_dag(G: nx.DiGraph, ax, title=""):
    """Nice layered layout based on topological order."""
    topo_order = list(nx.topological_sort(G))
    # assign y‑levels by topological rank
    layer = {n: i for i, n in enumerate(topo_order)}
    pos = {n: (n, -layer[n]) for n in G.nodes()}  # simple left‑to‑right

    nx.draw_networkx_nodes(G, pos, node_size=300, ax=ax)
    nx.draw_networkx_labels(
        G, pos, labels={n: G.nodes[n].get("name", str(n)) for n in G.nodes()}, ax=ax
    )

    # edge labels show tx / wc
    edge_labels = {
        (u, v): f"{G[u][v]['tx']}/{G[u][v]['wc']}" for u, v in G.edges()
    }

    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=12, ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    ax.set_title(title)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser(description="Random DAG playground")
    parser.add_argument("--nodes", type=int, default=10)
    parser.add_argument("--edges", type=int, default=30)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=None, help="Folder to save .graphml dumps"
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    graphs = []
    for idx in range(args.count):
        G = create_random_dag(args.nodes, args.edges, rng=rng)
        graphs.append(G)

        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            nx.write_graphml(G, args.out / f"dag_{idx}.graphml")

    if args.plot:
        n_cols = min(args.count, 3)
        n_rows = (args.count + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False
        )
        for ax, G, idx in zip(axes.flat, graphs, range(args.count)):
            draw_dag(G, ax, title=f"DAG #{idx}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
