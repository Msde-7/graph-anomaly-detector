from __future__ import annotations

from typing import Tuple, Dict, Any

import math
import random
import numpy as np
import pandas as pd
import networkx as nx

from graph_anomaly_detector.config import AppConfig


def _sample_cluster_size(avg_size: int) -> int:
    std = max(2, int(avg_size * 0.3))
    size = int(random.gauss(mu=avg_size, sigma=std))
    return max(3, size)


def generate_synthetic_graph(config: AppConfig) -> Tuple[nx.Graph, pd.DataFrame]:
    """Generate a synthetic social graph with human users and bot clusters.

    Returns a tuple of (Graph, metadata_df) where metadata has columns:
    - node: node id (int)
    - is_bot: bool
    - bot_cluster_id: int or -1 for humans
    """

    rng = np.random.default_rng(config.random_seed)
    random.seed(config.random_seed)

    human_count = config.num_humans

    # Human subgraph (Erdos-Renyi)
    human_graph = nx.fast_gnp_random_graph(human_count, config.human_edge_prob, seed=config.random_seed)

    G = nx.Graph()
    G.add_nodes_from(range(human_count))
    G.add_edges_from(human_graph.edges())

    meta_records = []
    for node_id in range(human_count):
        meta_records.append({"node": node_id, "is_bot": False, "bot_cluster_id": -1})

    next_node_id = human_count

    # Bot clusters
    for cluster_idx in range(config.num_bot_clusters):
        cluster_size = _sample_cluster_size(config.avg_bot_cluster_size)
        bot_nodes = list(range(next_node_id, next_node_id + cluster_size))
        next_node_id += cluster_size

        # Dense internal connections among bots in the same cluster
        internal = nx.fast_gnp_random_graph(cluster_size, config.bot_internal_edge_prob, seed=config.random_seed + cluster_idx + 1)
        internal = nx.relabel_nodes(internal, mapping={i: bot_nodes[i] for i in range(cluster_size)})

        G.add_nodes_from(bot_nodes)
        G.add_edges_from(internal.edges())

        # Sparse links from each bot to humans
        humans = list(range(human_count))
        for b in bot_nodes:
            # Number of human links ~ Binomial(human_count, p)
            k = rng.binomial(n=human_count, p=config.bot_to_human_edge_prob)
            if k > 0:
                targets = rng.choice(humans, size=min(k, human_count), replace=False)
                G.add_edges_from((b, int(h)) for h in targets)

        for b in bot_nodes:
            meta_records.append({"node": b, "is_bot": True, "bot_cluster_id": cluster_idx})

    meta_df = pd.DataFrame.from_records(meta_records)

    return G, meta_df
