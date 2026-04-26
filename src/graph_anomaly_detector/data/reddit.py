from __future__ import annotations

from typing import Tuple

import time
import pandas as pd
import networkx as nx

try:
    import praw  # type: ignore
except Exception:  # pragma: no cover
    praw = None


class RedditConfig:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent


def fetch_subreddit_interaction_graph(
    cfg: RedditConfig,
    subreddit: str,
    num_posts: int = 100,
    max_comments_per_post: int = 200,
    sleep_seconds: float = 0.0,
) -> Tuple[nx.Graph, pd.DataFrame]:
    """Build a user interaction graph for a subreddit.

    Nodes are authors (posters/commenters). Edges connect authors who interacted on the same post
    (post author ↔ commenter, and optional co-commenters). Attributes can be extended later.
    """
    if praw is None:
        raise RuntimeError("praw is not installed. Please install with `pip install praw`.")

    reddit = praw.Reddit(
        client_id=cfg.client_id,
        client_secret=cfg.client_secret,
        user_agent=cfg.user_agent,
    )

    G = nx.Graph()

    # Fetch posts
    sub = reddit.subreddit(subreddit)
    submissions = sub.new(limit=num_posts)

    user_set: set[str] = set()

    for submission in submissions:
        try:
            submission_author = getattr(submission.author, "name", None)
        except Exception:
            submission_author = None
        if submission_author is None:
            continue

        user_set.add(submission_author)
        G.add_node(submission_author)

        # Load comments
        submission.comments.replace_more(limit=0)
        count = 0
        for c in submission.comments.list():
            if count >= max_comments_per_post:
                break
            try:
                author = getattr(c.author, "name", None)
            except Exception:
                author = None
            if author is None:
                continue
            user_set.add(author)
            G.add_node(author)
            G.add_edge(submission_author, author)
            count += 1

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    meta_df = pd.DataFrame({"node": list(user_set)})
    # No ground truth labels; default is_bot to False
    meta_df["is_bot"] = False

    return G, meta_df
