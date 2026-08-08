from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import networkx as nx

from graph_anomaly_detector.config import AppConfig


def _sample_cluster_size(rng: np.random.Generator, avg_size: int) -> int:
    std = max(2, int(avg_size * 0.3))
    return max(3, int(rng.normal(loc=avg_size, scale=std)))


def generate_synthetic_graph(config: AppConfig) -> Tuple[nx.Graph, pd.DataFrame]:
    """Generate a synthetic social graph with human users and bot clusters.

    Returns a tuple of (Graph, metadata_df) where metadata has columns:
    - node: node id (int)
    - is_bot: bool
    - bot_cluster_id: int or -1 for humans
    """

    # A single local generator, so generating a graph leaves the global RNG untouched.
    rng = np.random.default_rng(config.random_seed)

    human_count = config.num_humans

    G = nx.fast_gnp_random_graph(human_count, config.human_edge_prob, seed=config.random_seed)

    meta_records = [
        {"node": node_id, "is_bot": False, "bot_cluster_id": -1}
        for node_id in range(human_count)
    ]

    next_node_id = human_count

    for cluster_idx in range(config.num_bot_clusters):
        cluster_size = _sample_cluster_size(rng, config.avg_bot_cluster_size)
        bot_nodes = list(range(next_node_id, next_node_id + cluster_size))
        next_node_id += cluster_size

        internal = nx.fast_gnp_random_graph(
            cluster_size,
            config.bot_internal_edge_prob,
            seed=config.random_seed + cluster_idx + 1,
        )
        relabel = {i: bot_nodes[i] for i in range(cluster_size)}
        G.add_nodes_from(bot_nodes)
        G.add_edges_from((relabel[u], relabel[v]) for u, v in internal.edges())

        for b in bot_nodes:
            # k is drawn from n=human_count trials, so it can never exceed the pool size.
            k = rng.binomial(n=human_count, p=config.bot_to_human_edge_prob)
            if k > 0:
                targets = rng.choice(human_count, size=k, replace=False)
                G.add_edges_from((b, int(h)) for h in targets)
            meta_records.append({"node": b, "is_bot": True, "bot_cluster_id": cluster_idx})

    meta_df = pd.DataFrame.from_records(meta_records)

    return G, meta_df
