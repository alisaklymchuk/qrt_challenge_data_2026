import sys
from src.config import load_config
from src.execution import execute

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    print(f"Loading configuration from: {config_path}")

    # Load configuration
    config = load_config(config_path)
    print("Configuration loaded successfully")

    # Execute the data loading and feature engineering pipeline
    print("Preparing data and building features...")
    prepared_data = execute(config)

    # Extract the prepared data
    dataset = prepared_data['dataset']  # Available for advanced usage (e.g., getting splits)
    X_train = prepared_data['X_train']
    y_train = prepared_data['y_train']
    X_test = prepared_data['X_test']
    feature_names = prepared_data['feature_names']

    # Example: you could use the dataset for cross-validation splits
    # splits = list(dataset.get_date_splits(n_splits=config.cv.n_splits))

    print(f"Data preparation completed!")
    print(f"  - Training features shape: {X_train.shape}")
    print(f"  - Test features shape: {X_test.shape}")
    print(f"  - Number of features: {len(feature_names)}")
    print(f"  - Target variable available: {y_train is not None}")
    if y_train is not None:
        print(f"  - Training target shape: {y_train.shape}")

    # Show first few feature names
    print(f"  - First 10 features: {feature_names[:10]}")

    # The user can now take X_train, y_train, X_test and pass them to their model
    # For example, to use the ridge model from src.models.ridge:
    # from src.models.ridge import build_ridge
    # model = build_ridge(X_train=X_train, y_train=y_train)
    # train_preds = model.predict(X_train)
    # test_preds = model.predict(X_test)

    print("\nPipeline completed successfully!")
    print("The data is now ready for model training.")
    print("To continue, you would:")
    print("1. Choose and instantiate your model")
    print("2. Train it on X_train, y_train")
    print("3. Make predictions on X_test")
    print("4. Save results to the results/ directory")

if __name__ == "__main__":
    main()