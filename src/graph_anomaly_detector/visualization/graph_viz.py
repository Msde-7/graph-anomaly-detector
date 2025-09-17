from __future__ import annotations

from typing import Iterable

import math
import json
import networkx as nx
import numpy as np
from pyvis.network import Network
import streamlit as st
import streamlit.components.v1 as components


def _scale(values: np.ndarray, min_size: float = 5.0, max_size: float = 20.0) -> np.ndarray:
    if values.size == 0:
        return values
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if math.isclose(vmin, vmax):
        return np.full_like(values, (min_size + max_size) / 2.0)
    return (values - vmin) / (vmax - vmin) * (max_size - min_size) + min_size


def render_pyvis(
    G: nx.Graph,
    scores: Iterable[float] | None = None,
    anomaly_flags: Iterable[bool] | None = None,
    height_px: int = 720,
) -> None:
    scores_arr = None if scores is None else np.asarray(list(scores), dtype=float)
    flags_arr = None if anomaly_flags is None else np.asarray(list(anomaly_flags), dtype=bool)

    degrees = np.array([deg for _, deg in G.degree()], dtype=float)
    sizes = _scale(degrees, 6.0, 22.0)

    net = Network(height=f"{height_px}px", width="100%", bgcolor="#0e1117", font_color="#eaecef")
    net.barnes_hut()

    # Base palette
    normal_color = "#8b949e"
    anomaly_color = "#ff4d4f"

    node_list = list(G.nodes())
    degree_map = dict(G.degree())

    for idx, n in enumerate(node_list):
        score = None if scores_arr is None else float(scores_arr[idx])
        is_anom = False if flags_arr is None else bool(flags_arr[idx])

        color = anomaly_color if is_anom else normal_color
        size = float(sizes[idx])
        title_parts = [f"node: {n}", f"degree: {degree_map[n]}"]
        if score is not None:
            title_parts.append(f"anomaly_score: {score:.4f}")
        if is_anom:
            title_parts.append("flag: anomaly")
        title = " | ".join(title_parts)

        net.add_node(
            n,
            label=str(n),
            title=title,
            color=color,
            size=size * (1.2 if is_anom else 1.0),
            shadow=is_anom,
            borderWidth=2 if is_anom else 0,
        )

    for u, v in G.edges():
        net.add_edge(u, v, color="#2f3542")

    options = {
        "nodes": {
            "font": {"color": "#eaecef"}
        },
        "edges": {
            "color": {"color": "#2f3542"},
            "smooth": False
        },
        "physics": {
            "stabilization": True,
            "barnesHut": {
                "gravitationalConstant": -20000,
                "springLength": 120,
                "springConstant": 0.04
            }
        },
        "interaction": {"hover": True, "tooltipDelay": 150, "hideEdgesOnDrag": False}
    }

    net.set_options(json.dumps(options))

    html = net.generate_html(notebook=False)
    components.html(html, height=height_px, scrolling=True)
