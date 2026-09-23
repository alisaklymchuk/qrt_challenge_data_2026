# Execution pipeline

from src.config import Config
from src.features import build_features
from src.models.model_utils import MODEL_REGISTRY, sign_acc_score

import pandas as pd

def execute(config):
    # Load data - assuming separate X and y files for train, combined for test
    train_x_df = pd.read_csv(config.data.train_path, index_col=0)
    test_df = pd.read_csv(config.data.test_path, index_col=0)

    # Load target if specified separately
    if hasattr(config.data, 'target_path') and config.data.target_path:
        y_train = pd.read_csv(config.data.target_path, index_col=0)['target']
        train_df = train_x_df.copy()
        train_df['target'] = y_train
    else:
        # Try to extract target from train data
        y_train = train_x_df['target'] if 'target' in train_x_df.columns else None
        train_df = train_x_df.copy()
        if y_train is not None:
            # Target is already in the dataframe
            pass
        else:
            # No target available
            y_train = None

    # Build features (excluding target column if present)
    feature_cols = [col for col in train_df.columns if col != 'target']
    train_features = build_features(train_df[feature_cols], config.features)
    test_features = build_features(test_df, config.features)

    # Train model
    model = MODEL_REGISTRY[config.model.name](X_train=train_features, y_train=y_train, **config.model.params)

    # Make predictions
    train_preds = model.predict(train_features)
    test_preds = model.predict(test_features)

    # Calculate score if we have labels
    score = None
    if y_train is not None:
        score = sign_acc_score(y_train, train_preds)

    return {
        'train_predictions': train_preds,
        'test_predictions': test_preds,
        'score': score
    }