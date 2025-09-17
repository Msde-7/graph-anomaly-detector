from pydantic import BaseModel, Field

class AppConfig(BaseModel):
    """Configuration for synthetic graph generation and model training."""

    random_seed: int = Field(42, description="Random seed for reproducibility")
    num_humans: int = Field(800, ge=1, description="Number of human accounts")
    num_bot_clusters: int = Field(3, ge=1, description="Number of bot clusters")
    avg_bot_cluster_size: int = Field(30, ge=2, description="Average size of each bot cluster")

    human_edge_prob: float = Field(0.004, ge=0.0, le=1.0, description="Edge probability among humans")
    bot_internal_edge_prob: float = Field(0.35, ge=0.0, le=1.0, description="Edge probability within bot clusters")
    bot_to_human_edge_prob: float = Field(0.005, ge=0.0, le=1.0, description="Edge probability from bot to human")

    contamination: float = Field(0.08, ge=0.0, le=0.5, description="Expected fraction of anomalies")
    n_estimators: int = Field(300, ge=50, description="IsolationForest number of trees")
