from __future__ import annotations

import community as community_louvain  # type: ignore
import networkx as nx
import pandas as pd


def compute_node_features(G: nx.Graph) -> pd.DataFrame:
    """Compute graph features per node for anomaly detection."""
    n = G.number_of_nodes()

    degree_dict = dict(G.degree())
    clustering = nx.clustering(G)
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)

    # Approximate betweenness for speed on large graphs; k must not exceed n.
    k_samples = min(n, max(10, n // 10), 200) if n > 0 else 0
    betweenness = (
        nx.betweenness_centrality(G, k=k_samples, normalized=True, seed=42)
        if k_samples > 0
        else {}
    )

    community_ids = community_louvain.best_partition(G) if n > 0 else {}

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
