from __future__ import annotations

from typing import Dict

import networkx as nx
import pandas as pd

try:
    import community as community_louvain  # type: ignore
except Exception:  # pragma: no cover
    community_louvain = None  # fallback if package missing


def compute_node_features(G: nx.Graph) -> pd.DataFrame:
    """Compute graph features per node for anomaly detection."""
    nodes = list(G.nodes())

    degree_dict = dict(G.degree())
    clustering = nx.clustering(G)

    # PageRank is robust for social graphs
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)

    # Approximate betweenness for speed on large graphs
    k_samples = min(200, max(10, int(len(nodes) * 0.1)))
    try:
        betweenness = nx.betweenness_centrality(G, k=k_samples, normalized=True, seed=42)
    except TypeError:
        # Older networkx without seed param
        betweenness = nx.betweenness_centrality(G, k=k_samples, normalized=True)

    # Communities (Louvain) if available
    community_ids: Dict[int, int]
    if community_louvain is not None and len(nodes) > 0:
        community_ids = community_louvain.best_partition(G)
    else:
        community_ids = {n: 0 for n in nodes}

    df = pd.DataFrame(
        {
            "degree": pd.Series(degree_dict),
            "clustering_coef": pd.Series(clustering),
            "pagerank": pd.Series(pagerank),
            "betweenness": pd.Series(betweenness),
            "community_id": pd.Series(community_ids),
        }
    ).sort_index()

    return df
