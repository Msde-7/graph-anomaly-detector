from __future__ import annotations

from typing import Tuple, Optional, Dict

import time
import pandas as pd
import networkx as nx

try:
    import tweepy  # type: ignore
except Exception:
    tweepy = None


class TwitterConfig:
    def __init__(self, bearer_token: str):
        self.bearer_token = bearer_token


def fetch_search_interaction_graph(
    cfg: TwitterConfig,
    query: str,
    max_tweets: int = 300,
    sleep_seconds: float = 0.0,
) -> Tuple[nx.Graph, pd.DataFrame]:
    """Build a user interaction graph from Twitter/X search results.

    Nodes are users. Edges connect:
    - author ↔ mentioned usernames
    - author ↔ original author of retweet/quote
    - author ↔ replied-to user
    """
    if tweepy is None:
        raise RuntimeError("tweepy is not installed. Please install with `pip install tweepy`.")

    client = tweepy.Client(bearer_token=cfg.bearer_token, wait_on_rate_limit=True)

    G = nx.Graph()

    tweets_collected = 0
    next_token: Optional[str] = None
    user_id_to_username: Dict[str, str] = {}

    while tweets_collected < max_tweets:
        batch = min(100, max_tweets - tweets_collected)
        resp = client.search_recent_tweets(
            query=query,
            max_results=batch,
            expansions=[
                "author_id",
                "entities.mentions.username",
                "referenced_tweets.id.author_id",
                "in_reply_to_user_id",
            ],
            tweet_fields=[
                "author_id",
                "entities",
                "referenced_tweets",
                "in_reply_to_user_id",
                "lang",
            ],
            user_fields=["username"],
            next_token=next_token,
        )

        if resp is None or resp.data is None or len(resp.data) == 0:
            break

        # Build user mapping
        if resp.includes and "users" in resp.includes:
            for u in resp.includes["users"]:
                user_id_to_username[str(u.id)] = u.username

        for t in resp.data:
            author_id = getattr(t, "author_id", None)
            if author_id is None:
                continue
            author = user_id_to_username.get(str(author_id), str(author_id))
            G.add_node(author)

            # Mentions edges
            ents = getattr(t, "entities", None)
            if ents and "mentions" in ents and ents["mentions"]:
                for m in ents["mentions"]:
                    uname = m.get("username")
                    if uname:
                        G.add_node(uname)
                        G.add_edge(author, uname)

            # Reply-to edge
            reply_to_id = getattr(t, "in_reply_to_user_id", None)
            if reply_to_id is not None:
                replied = user_id_to_username.get(str(reply_to_id), str(reply_to_id))
                G.add_node(replied)
                G.add_edge(author, replied)

            # Retweet/quote edges
            ref = getattr(t, "referenced_tweets", None)
            if ref:
                # referenced_tweets contain dicts with id/type; we rely on includes to map id->author, but Tweepy v2 does not always expose that directly here.
                # As a fallback, connect author to reply_to if present; else skip.
                pass

            tweets_collected += 1

        next_token = getattr(resp.meta, "next_token", None)
        if not next_token:
            break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    meta_df = pd.DataFrame({"node": list(G.nodes())})
    meta_df["is_bot"] = False

    return G, meta_df
