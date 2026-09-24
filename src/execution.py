# Execution pipeline - updated to use Dataset class for i.i.d. data
# Realized that dates are anonymized and shuffled, so no temporal dependence exists.
# This version prepares the data and features but does NOT train the model
# The user can then take the prepared data and pass it to their model of choice

from src.dataset import QuantDataset


def execute(config):
    """
    Execute the data loading and feature engineering pipeline.

    This function:
    1. Loads the data using the QuantDataset class
    2. Builds features according to the configuration
    3. Splits data into train and eval sets using random sampling (appropriate for i.i.d. data)

    Args:
        config: Configuration object

    Returns:
        None - function is incomplete, user will continue writing
    """
    # Initialize and load data using the Dataset class
    dataset = QuantDataset(config)
    dataset.load_data()

    # Build features using the default feature set from config
    X_train, X_test, feature_names = dataset.build_feature_set(
        feature_set_name="default"
    )

    # Get the target variable
    y_train = dataset.y_train

    # Split data into train and eval sets using random sampling
    # Appropriate for i.i.d. data (since dates are anonymized/shuffled)
    X_train_final, X_eval, y_train_final, y_eval = dataset.get_train_val_split(
        test_size=0.2,  # 20% for evaluation
        shuffle=True,
        random_state=config.cv.random_state
    )
    