import sys
from pathlib import Path

# Ensure src is on path for local imports
sys.path.append(str(Path(__file__).resolve().parent / "src"))

import streamlit as st
import pandas as pd

from graph_anomaly_detector.config import AppConfig
from graph_anomaly_detector.data.synthetic import generate_synthetic_graph
from graph_anomaly_detector.data.ingest import load_graph_from_edge_csv
from graph_anomaly_detector.data.reddit import fetch_subreddit_interaction_graph, RedditConfig
from graph_anomaly_detector.data.twitter import fetch_search_interaction_graph, TwitterConfig
from graph_anomaly_detector.features.graph_features import compute_node_features
from graph_anomaly_detector.models.isolation_forest import fit_and_score
from graph_anomaly_detector.visualization.graph_viz import render_pyvis

st.set_page_config(page_title="Graph Anomaly Detector", layout="wide")

st.title("Graph Anomaly Detector 🔍")
st.caption("Synthetic / CSV / Reddit / Twitter → detect spam/bot clusters → visualize anomalies")

with st.sidebar:
    st.header("Data Source")
    data_source = st.radio("Choose data source", ["Synthetic", "CSV Upload", "Reddit API", "Twitter API"], horizontal=False)

    reddit_inputs = {}
    twitter_inputs = {}
    if data_source == "Reddit API":
        st.caption("Enter Reddit API credentials. Create an app at https://www.reddit.com/prefs/apps")
        reddit_client_id = st.text_input("Client ID", value="", type="default")
        reddit_client_secret = st.text_input("Client Secret", value="", type="password")
        reddit_user_agent = st.text_input("User Agent", value="graph-anomaly-detector/0.1 by YOUR_REDDIT_USERNAME")
        subreddit = st.text_input("Subreddit (without r/)", value="technology")
        num_posts = st.slider("Posts to fetch", min_value=10, max_value=500, value=100, step=10)
        max_comments = st.slider("Max comments per post", min_value=20, max_value=500, value=200, step=20)
        reddit_inputs = {
            "client_id": reddit_client_id,
            "client_secret": reddit_client_secret,
            "user_agent": reddit_user_agent,
            "subreddit": subreddit,
            "num_posts": num_posts,
            "max_comments": max_comments,
        }
    elif data_source == "Twitter API":
        st.caption("Enter a Twitter/X Bearer Token (Academic/Elevated recommended).")
        bearer = st.text_input("Bearer Token", value="", type="password")
        query = st.text_input("Search Query", value="chatgpt lang:en -is:retweet")
        max_tweets = st.slider("Max tweets", min_value=50, max_value=1000, value=300, step=50)
        twitter_inputs = {
            "bearer": bearer,
            "query": query,
            "max_tweets": max_tweets,
        }

    st.divider()
    st.header("Configuration")
    config = AppConfig(
        random_seed=st.number_input("Random seed", value=42, step=1),
        num_humans=st.slider("Number of human accounts", min_value=200, max_value=5000, value=800, step=50),
        num_bot_clusters=st.slider("Number of bot clusters", min_value=1, max_value=10, value=3, step=1),
        avg_bot_cluster_size=st.slider("Avg bot cluster size", min_value=5, max_value=200, value=30, step=5),
        human_edge_prob=st.slider("Human-human edge prob", min_value=0.0005, max_value=0.02, value=0.004, step=0.0005),
        bot_internal_edge_prob=st.slider("Bot internal edge prob", min_value=0.05, max_value=1.0, value=0.35, step=0.05),
        bot_to_human_edge_prob=st.slider("Bot→human edge prob", min_value=0.0, max_value=0.05, value=0.005, step=0.001),
        contamination=st.slider("Model contamination (expected anomaly rate)", min_value=0.01, max_value=0.25, value=0.08, step=0.01),
        n_estimators=st.slider("IsolationForest trees", min_value=100, max_value=1000, value=300, step=50),
    )

    if data_source == "CSV Upload":
        st.caption("Upload an edges CSV (columns: source,target). Optional: nodes CSV with 'node' and attributes like 'is_bot'.")
        uploaded_edges = st.file_uploader("Edges CSV", type=["csv"], key="edges")
        uploaded_nodes = st.file_uploader("Nodes CSV (optional)", type=["csv"], key="nodes")

    run = st.button("Generate/Load & Detect", type="primary")

@st.cache_data(show_spinner=False)
def _generate_graph_cached(_cfg: AppConfig):
    G, meta_df = generate_synthetic_graph(_cfg)
    return G, meta_df

@st.cache_data(show_spinner=False)
def _compute_features_cached(nodes, edges):
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    features = compute_node_features(G)
    return features

@st.cache_data(show_spinner=False)
def _load_csv_cached(edges_bytes: bytes, nodes_bytes: bytes | None, edges_name: str, nodes_name: str | None):
    import io
    edges_df = pd.read_csv(io.BytesIO(edges_bytes))
    nodes_df = None
    if nodes_bytes is not None:
        nodes_df = pd.read_csv(io.BytesIO(nodes_bytes))
    G, meta_df = load_graph_from_edge_csv(edges_df, nodes_df)
    return G, meta_df

if run:
    if data_source == "Synthetic":
        with st.spinner("Generating graph..."):
            G, meta_df = _generate_graph_cached(config)
    elif data_source == "CSV Upload":
        if uploaded_edges is None:
            st.error("Please upload an edges CSV.")
            st.stop()
        with st.spinner("Loading CSV graph..."):
            edges_bytes = uploaded_edges.getvalue()
            nodes_bytes = uploaded_nodes.getvalue() if uploaded_nodes is not None else None
            G, meta_df = _load_csv_cached(edges_bytes, nodes_bytes, uploaded_edges.name, uploaded_nodes.name if uploaded_nodes else None)
    elif data_source == "Reddit API":
        missing = [k for k, v in reddit_inputs.items() if k in ("client_id","client_secret","user_agent") and not v]
        if missing:
            st.error("Please enter Reddit API credentials: Client ID, Client Secret, User Agent.")
            st.stop()
        with st.spinner("Fetching subreddit interactions..."):
            rcfg = RedditConfig(reddit_inputs["client_id"], reddit_inputs["client_secret"], reddit_inputs["user_agent"])
            G, meta_df = fetch_subreddit_interaction_graph(
                cfg=rcfg,
                subreddit=reddit_inputs["subreddit"],
                num_posts=int(reddit_inputs["num_posts"]),
                max_comments_per_post=int(reddit_inputs["max_comments"]),
                sleep_seconds=0.0,
            )
    else:  # Twitter API
        if not twitter_inputs.get("bearer"):
            st.error("Please enter a Twitter/X Bearer Token.")
            st.stop()
        with st.spinner("Fetching tweets and building graph..."):
            tcfg = TwitterConfig(twitter_inputs["bearer"])
            G, meta_df = fetch_search_interaction_graph(
                cfg=tcfg,
                query=twitter_inputs["query"],
                max_tweets=int(twitter_inputs["max_tweets"]),
                sleep_seconds=0.0,
            )

    st.write(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")

    with st.spinner("Computing graph features..."):
        features = _compute_features_cached(list(G.nodes()), list(G.edges()))

    with st.spinner("Training anomaly detector..."):
        model, scores, anomaly_labels = fit_and_score(
            features,
            contamination=config.contamination,
            n_estimators=config.n_estimators,
            random_state=config.random_seed,
        )

    results = (
        pd.DataFrame({
            "node": features.index,
            "anomaly_score": scores,
            "is_anomaly": anomaly_labels.astype(int),
        })
        .set_index("node")
        .join(features)
        .join(meta_df.set_index("node"))
    )

    # Align scores/flags to graph node iteration order
    node_order = list(G.nodes())
    scores_aligned = results.loc[node_order, "anomaly_score"].values
    flags_aligned = results.loc[node_order, "is_anomaly"].astype(bool).values

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Interactive Graph")
        render_pyvis(G, scores=scores_aligned, anomaly_flags=flags_aligned)

    with right:
        st.subheader("Top Suspicious Accounts")
        top_k = st.slider("Show top K", min_value=5, max_value=100, value=20, step=5)
        top_df = results.sort_values("anomaly_score", ascending=False).head(top_k)
        cols = ["anomaly_score", "is_anomaly", "degree", "pagerank", "clustering_coef", "community_id"]
        if "is_bot" in results.columns:
            cols.insert(2, "is_bot")
        st.dataframe(top_df[cols])

        if "is_bot" in results.columns:
            from sklearn.metrics import precision_recall_fscore_support
            p, r, f1, _ = precision_recall_fscore_support(
                results["is_bot"].astype(int),
                results["is_anomaly"].astype(int),
                average="binary",
            )
            st.markdown(f"**Precision**: {p:.2f} | **Recall**: {r:.2f} | **F1**: {f1:.2f}")

        st.download_button(
            "Download results CSV",
            data=results.reset_index().to_csv(index=False),
            file_name="anomaly_results.csv",
            mime="text/csv",
        )
else:
    st.info("Choose data source and configuration, then click Generate/Load & Detect.")
