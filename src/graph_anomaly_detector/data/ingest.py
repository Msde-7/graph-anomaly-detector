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

    # A row with a blank endpoint would otherwise become a node named "nan" or "None".
    edges = edge_df[[source_col, target_col]].dropna()

    # Normalize node identifiers to strings to support mixed ID types
    edges = edges.astype(str)

    # Self-loops inflate degree without describing an interaction between two accounts.
    edges = edges[edges[source_col] != edges[target_col]]

    # Build graph
    G = nx.Graph()
    G.add_edges_from(zip(edges[source_col], edges[target_col]))

    # Build metadata
    nodes = list(G.nodes())
    if node_df is not None and "node" in node_df.columns:
        node_meta = node_df.copy()
        node_meta["node"] = node_meta["node"].astype(str)
        # Repeated rows for one node would duplicate metadata and misalign the results
        # frame against graph iteration order when the graph is rendered.
        node_meta = node_meta.drop_duplicates(subset="node", keep="first")
        meta_df = pd.DataFrame({"node": nodes}).merge(node_meta, on="node", how="left")
    else:
        meta_df = pd.DataFrame({"node": nodes})

    # Nodes missing from the nodes CSV arrive as NaN, which cannot be cast to int later.
    # where() rather than fillna() keeps pandas from warning about downcasting.
    if "is_bot" in meta_df.columns:
        labels = meta_df["is_bot"]
        meta_df["is_bot"] = labels.where(labels.notna(), False).astype(bool)
    else:
        meta_df["is_bot"] = False

    return G, meta_df
