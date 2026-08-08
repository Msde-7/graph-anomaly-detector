from __future__ import annotations

import community as community_louvain  # type: ignore
import networkx as nx
import pandas as pd

# Columns the anomaly model trains on. community_id is deliberately left out because it is
# a nominal label, so its numeric value carries no ordering the model could learn from.
# Including it makes the flagged set depend on arbitrary community numbering.
MODEL_FEATURE_COLUMNS = ("degree", "clustering_coef", "pagerank", "betweenness")

ALL_FEATURE_COLUMNS = (*MODEL_FEATURE_COLUMNS, "community_id")


def compute_node_features(G: nx.Graph, seed: int = 42) -> pd.DataFrame:
    """Compute graph features per node for anomaly detection.

    Returns one row per node, indexed by node id. community_id is included for display
    and grouping but is not part of MODEL_FEATURE_COLUMNS.
    """
    n = G.number_of_nodes()
    if n == 0:
        return pd.DataFrame(columns=list(ALL_FEATURE_COLUMNS))

    # Approximate betweenness keeps large graphs fast. k must not exceed the node count.
    k_samples = min(n, max(10, n // 10), 200)

    # python-louvain falls back to numpy's global RandomState when random_state is None,
    # which makes community ids vary between runs even with the seed pinned here.
    community_ids = community_louvain.best_partition(G, random_state=seed)

    return pd.DataFrame(
        {
            "degree": pd.Series(dict(G.degree())),
            "clustering_coef": pd.Series(nx.clustering(G)),
            "pagerank": pd.Series(nx.pagerank(G, alpha=0.85, max_iter=100)),
            "betweenness": pd.Series(
                nx.betweenness_centrality(G, k=k_samples, normalized=True, seed=seed)
            ),
            "community_id": pd.Series(community_ids),
        }
    ).sort_index()
