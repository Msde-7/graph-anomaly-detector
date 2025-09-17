from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd
import networkx as nx


def load_graph_from_edge_csv(
    edge_df: pd.DataFrame,
    node_df: Optional[pd.DataFrame] = None,
    source_col: str = "source",
    target_col: str = "target",
) -> Tuple[nx.Graph, pd.DataFrame]:
    """Create a graph from an edge list CSV and optional node metadata CSV.

    edge_df must contain columns [source_col, target_col].
    node_df, if provided, should have a 'node' column and optional attributes such as 'is_bot'.
    """
    if source_col not in edge_df.columns or target_col not in edge_df.columns:
        raise ValueError(f"Edges CSV must include '{source_col}' and '{target_col}' columns")

    # Normalize node identifiers to strings to support mixed ID types
    edge_df = edge_df.copy()
    edge_df[source_col] = edge_df[source_col].astype(str)
    edge_df[target_col] = edge_df[target_col].astype(str)

    # Build graph
    G = nx.Graph()
    edges = list(zip(edge_df[source_col], edge_df[target_col]))
    G.add_edges_from(edges)

    # Build metadata
    nodes = list(G.nodes())
    if node_df is not None and "node" in node_df.columns:
        node_meta = node_df.copy()
        node_meta["node"] = node_meta["node"].astype(str)
        meta_df = pd.DataFrame({"node": nodes}).merge(node_meta, on="node", how="left")
    else:
        meta_df = pd.DataFrame({"node": nodes})

    # Default 'is_bot' to False if missing
    if "is_bot" not in meta_df.columns:
        meta_df["is_bot"] = False

    return G, meta_df
