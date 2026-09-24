"""
Example showing how to continue from where the execution pipeline stops.

This demonstrates how a user would take the prepared data and:
1. Train a model (e.g., ridge regression)
2. Make predictions
3. Store results in the results/ directory
"""


import os
import pandas as pd
from src.config import load_config
from src.execution import execute
from src.models.ridge import build_ridge
from src.models.model_utils import sign_acc_score


def main():
    # Step 1: Use our execution pipeline to prepare data and features
    print("=== Step 1: Prepare data and features ===")
    config = load_config('configs/default.yaml')
    prepared_data = execute(config)

    # Extract the prepared data
    X_train = prepared_data['X_train']
    y_train = prepared_data['y_train']
    X_test = prepared_data['X_test']
    feature_names = prepared_data['feature_names']
    dataset = prepared_data['dataset']  # Available for advanced usage

    print(f"Training data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Features: {len(feature_names)}")

    # Step 2: Handle missing values (if any) before modeling
    # This is an example of preprocessing the user might need to do
    print("\n=== Step 2: Preprocess data ===")
    # Check for NaN values
    nan_count_train = X_train.isna().sum().sum()
    nan_count_test = X_test.isna().sum().sum()
    print(f"NaN values in training features: {nan_count_train}")
    print(f"NaN values in test features: {nan_count_test}")

    # Fill NaN values with 0 (simple approach - in practice, users might choose different strategies)
    X_train_filled = X_train.fillna(0)
    X_test_filled = X_test.fillna(0)

    if nan_count_train > 0 or nan_count_test > 0:
        print("Filled NaN values with 0 for modeling")
    else:
        X_train_filled = X_train
        X_test_filled = X_test
        print("No NaN values found - using original data")

    # Step 3: Train a model
    print("\n=== Step 3: Train model ===")
    model = build_ridge(X_train=X_train_filled, y_train=y_train)
    print("Model trained successfully!")

    # Step 4: Make predictions
    print("\n=== Step 4: Make predictions ===")
    train_preds = model.predict(X_train_filled)
    test_preds = model.predict(X_test_filled)

    # Step 5: Evaluate (if we have labels)
    print("\n=== Step 5: Evaluate ===")
    if y_train is not None:
        # For direct function use, use sign_acc; for scorer, need model and X
        from src.models.model_utils import sign_acc
        train_score = sign_acc(y_train, train_preds)
        print(f"Training accuracy: {train_score:.4f}")

    # Step 6: Save results
    print("\n=== Step 6: Save results ===")

    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Save predictions
    train_pred_df = pd.DataFrame({'prediction': train_preds}, index=X_train.index)
    train_pred_path = os.path.join(results_dir, "train_predictions.csv")
    train_pred_df.to_csv(train_pred_path)

    test_pred_df = pd.DataFrame({'prediction': test_preds}, index=X_test.index)
    test_pred_path = os.path.join(results_dir, "test_predictions.csv")
    test_pred_df.to_csv(test_pred_path)

    # Save feature names
    feature_path = os.path.join(results_dir, "feature_names.txt")
    with open(feature_path, 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")

    # Save model parameters/info (simplified)
    model_info_path = os.path.join(results_dir, "model_info.txt")
    with open(model_info_path, 'w') as f:
        f.write(f"Model type: Ridge Regression with GridSearchCV\n")
        f.write(f"Best alpha: {model.named_steps['ridge'].alpha_}\n")
        f.write(f"Number of features: {len(feature_names)}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")

    print(f"Results saved to {results_dir}/:")
    print(f"  - train_predictions.csv")
    print(f"  - test_predictions.csv")
    print(f"  - feature_names.txt")
    print(f"  - model_info.txt")

    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()