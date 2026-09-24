# Execution pipeline - updated to use Dataset class for i.i.d. data
# Realized that dates are anonymized and shuffled, so no temporal dependence exists.
# This version prepares the data and features but does NOT train the model
# The user can then take the prepared data and pass it to their model of choice

from src.dataset import QuantDataset
from src.models.model_utils import MODEL_REGISTRY, sign_acc_score, sign_acc
from datetime import datetime
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

def execute(config):
    """
    Execute the data loading, feature engineering, modeling, and results saving pipeline.

    This function:
    1. Loads the data using the QuantDataset class
    2. Builds features according to the configuration
    3. Splits data into train and eval sets using random sampling (appropriate for i.i.d. data)
    4. Trains a model and saves results, config, and predictions

    Args:
        config: Configuration object

    Returns:
        dict: Contains dataset, X_train, y_train, X_test, and feature_names for further use
    """
    # Initialize and load data using the Dataset class
    dataset = QuantDataset(config)
    dataset.load_data()

    # Build features using the default feature set from config
    X_train_features, X_test_features, feature_names = dataset.build_feature_set(
        feature_set_name="default"
    )

    # Handle missing values by filling with 0 (as NaN values cause issues with Ridge model)
    X_train_features = X_train_features.fillna(0)
    X_test_features = X_test_features.fillna(0)

    # Get the target variable
    y_train = dataset.y_train

    # Split FEATURES data into train and eval sets using random sampling
    # Appropriate for i.i.d. data (since dates are anonymized/shuffled)
    X_train, X_eval, y_train, y_eval = train_test_split(
        X_train_features, y_train,
        test_size=0.2,  # 20% for evaluation
        shuffle=True,
        random_state=config.cv.random_state
    )
    # Assumes the model is already trained
    model = MODEL_REGISTRY[config.model.model_name](X_train, y_train, **config.model.params)
    y_train_pred, y_eval_pred = model.predict(X_train), model.predict(X_eval)
    y_test_pred = model.predict(X_test_features)
    train_acc, eval_acc = sign_acc(y_train, y_train_pred), sign_acc(y_eval, y_eval_pred)

    # Create folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = config.experiment.name
    experiment_dir = os.path.join("results", f"{experiment_name}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # Save config (simply dump the config object to YAML)
    config_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save predictions (train, eval, and test)
    train_predictions_df = pd.DataFrame({'prediction': y_train_pred}, index=X_train.index)
    train_predictions_path = os.path.join(experiment_dir, "train_predictions.csv")
    train_predictions_df.to_csv(train_predictions_path)

    eval_predictions_df = pd.DataFrame({'prediction': y_eval_pred}, index=X_eval.index)
    eval_predictions_path = os.path.join(experiment_dir, "eval_predictions.csv")
    eval_predictions_df.to_csv(eval_predictions_path)

    test_predictions_df = pd.DataFrame({'prediction': y_test_pred}, index=X_test_features.index)
    test_predictions_path = os.path.join(experiment_dir, "test_predictions.csv")
    test_predictions_df.to_csv(test_predictions_path)

    # Save results (metrics and info)
    results_path = os.path.join(experiment_dir, "results.json")
    results_data = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "train_accuracy": float(train_acc),
        "eval_accuracy": float(eval_acc),
        "n_train_samples": len(X_train),
        "n_eval_samples": len(X_eval),
        "n_features": len(feature_names),
        "feature_names": feature_names
    }
    import json
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)