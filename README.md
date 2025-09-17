# Graph Anomaly Detector

Build a user/content graph from synthetic data, CSV files, or live APIs (Reddit, Twitter/X). Compute graph features and run a simple anomaly detector to surface suspicious clusters, then explore results in an interactive graph view.

## Features
- Data sources: Synthetic generator, CSV upload, Reddit API, Twitter/X API
- Features: degree, clustering coefficient, PageRank, approximate betweenness, Louvain community id
- Anomaly detection: IsolationForest with tunable contamination and number of trees
- Visualization: interactive PyVis network; anomalies highlighted; CSV export of results
- Evaluation (optional): if `is_bot` labels exist in nodes data, compute precision/recall/F1

## Install
```bash
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
```bash
streamlit run app.py
```

## Data sources
- Synthetic: generates humans plus dense bot-like clusters
- CSV Upload: edge list and optional node metadata
- Reddit API: user interaction graph for a subreddit (author ↔ commenters)
- Twitter/X API: user interaction graph from a search query (author ↔ mentions/replies)

### CSV schema
- Edges CSV (required): columns `source,target`
- Nodes CSV (optional): column `node` plus any attributes (e.g., `is_bot`)

### Reddit API setup
1. Create an app at `https://www.reddit.com/prefs/apps`
2. Copy client id, client secret, and define a user agent (e.g., `graph-anomaly-detector/0.1 by <username>`) 
3. In the app sidebar, choose Reddit API, paste credentials, set subreddit and limits, then run

### Twitter/X API setup
1. Obtain a Twitter API v2 Bearer Token
2. In the app sidebar, choose Twitter API, paste the token
3. Enter a search query (e.g., `chatgpt lang:en -is:retweet`) and a max tweets value, then run

## Notes
- IDs are treated as strings internally (numeric or text are fine)
- API data does not include ground-truth labels; metrics require a Nodes CSV with `is_bot`
- Large graphs can be slow to render; reduce node count or use stricter filters

## How it works (brief)
1. Load/build a NetworkX graph
2. Compute node features
3. Train IsolationForest and score nodes
4. Render graph and surface top anomalies
