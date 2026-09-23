# Execution pipeline

from src.config import Config
from src.features import build_features
from src.models import train_model, evaluate_model

def execute(config):
    # Load data
    train_df = pd.read_csv(config.data.train_path, index_col=0)
    test_df = pd.read_csv(config.data.test_path, index_col=0)

    # Build features
    train_features = build_features(train_df, config.features)
    test_features = build_features(test_df, config.features)

    # Train model
    model = train_model(train_features, train_df["target"], config.model)

    # Evaluate model
    metrics = evaluate_model(model, test_features, test_df["target"])

    return metrics