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
        # search_recent_tweets only accepts max_results between 10 and 100. Pages often come
        # back short, so the remaining count can fall under 10 and would be rejected.
        batch = max(10, min(100, max_tweets - tweets_collected))
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

        # The referenced_tweets.id.author_id expansion returns the retweeted and quoted
        # tweets themselves, which is how a retweet is traced back to its original author.
        referenced_authors: Dict[str, str] = {}
        if resp.includes and "tweets" in resp.includes:
            for rt in resp.includes["tweets"]:
                rt_author = getattr(rt, "author_id", None)
                if rt_author is not None:
                    referenced_authors[str(rt.id)] = str(rt_author)

        for t in resp.data:
            author_id = getattr(t, "author_id", None)
            if author_id is None:
                continue
            author = user_id_to_username.get(str(author_id), str(author_id))
            G.add_node(author)

            counterparts = []

            # Mentions
            ents = getattr(t, "entities", None)
            if ents and ents.get("mentions"):
                counterparts.extend(m.get("username") for m in ents["mentions"])

            # Reply-to
            reply_to_id = getattr(t, "in_reply_to_user_id", None)
            if reply_to_id is not None:
                counterparts.append(user_id_to_username.get(str(reply_to_id), str(reply_to_id)))

            # Retweet or quote of another account
            for ref in getattr(t, "referenced_tweets", None) or []:
                ref_author_id = referenced_authors.get(str(getattr(ref, "id", "")))
                if ref_author_id is not None:
                    counterparts.append(user_id_to_username.get(ref_author_id, ref_author_id))

            for counterpart in counterparts:
                # Mentioning or replying to yourself is not an interaction between two
                # accounts, so skip it rather than adding a degree-inflating self-loop.
                if counterpart and counterpart != author:
                    G.add_node(counterpart)
                    G.add_edge(author, counterpart)

            tweets_collected += 1

        next_token = getattr(resp.meta, "next_token", None)
        if not next_token:
            break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    meta_df = pd.DataFrame({"node": list(G.nodes())})
    meta_df["is_bot"] = False

    return G, meta_df
