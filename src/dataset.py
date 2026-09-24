"""
Dataset class for handling financial data as independent and identically distributed (i.i.d.) samples.
Realized that dates are anonymized and shuffled, so no temporal dependence exists.
"""

import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from typing import Tuple, List, Optional
from .features import get_base_feature_columns, apply_feature, build_features
from .config import Config


class QuantDataset:
    """
    A dataset class for handling financial data as independent samples.

    Important realization: Dates are anonymized and shuffled, so there is no temporal dependence.
    Each row represents an independent observation (sample) and should be treated as such.

    This class encapsulates:
    - Loading of separate X_train.csv, X_test.csv, y_train.csv files
    - Feature engineering using configurable transformations
    - Standard random train/validation/test split generation (appropriate for i.i.d. data)
    - Easy access to data for model training and evaluation
    """

    def __init__(self, config: Config):
        """
        Initialize the dataset with configuration.

        Args:
            config: Configuration object containing data and experiment settings
        """
        self.config = config
        self.raw_train = None
        self.raw_test = None
        self.y_train = None
        self.train_features = None
        self.test_features = None
        self.feature_names = None

        # Cache for different feature configurations to avoid recomputation
        self._feature_cache = {}

    def load_data(self) -> None:
        """
        Load the raw data files.

        Expects:
        - X_train.csv: features for training (with ROW_ID as index)
        - X_test.csv: features for testing (with ROW_ID as index)
        - y_train.csv: target for training (with ROW_ID as index, containing 'target' column)

        Note: Since dates are anonymized and shuffled, we treat all rows as independent samples.
        """
        # Load feature data
        self.raw_train = pd.read_csv(
            self.config.data.train_path,
            index_col='ROW_ID'
        )
        self.raw_test = pd.read_csv(
            self.config.data.test_path,
            index_col='ROW_ID'
        )

        # Load target data
        self.y_train = pd.read_csv(
            self.config.data.target_path,
            index_col='ROW_ID'
        )['target']

    def get_train_test_split(self,
                           test_size: float = 0.2,
                           shuffle: bool = True,
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Split the training data into train and validation sets using random sampling.
        Appropriate for i.i.d. data (since dates are anonymized/shuffled).

        Args:
            test_size: Proportion of dataset to include in validation split
            shuffle: Whether to shuffle before splitting
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        if self.raw_train is None or self.y_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Use scikit-learn's train_test_split for i.i.d. data
        X_train, X_val, y_train, y_val = train_test_split(
            self.raw_train,
            self.y_train,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state
        )

        return X_train, y_train, X_val, y_val

    def build_feature_set(self,
                         feature_set_name: str = "default",
                         force_rebuild: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Build feature matrices for training and test data.

        Args:
            feature_set_name: Name identifier for the feature set (for caching)
            force_rebuild: If True, rebuild features even if cached

        Returns:
            Tuple of (train_features_df, test_features_df, feature_names_list)
        """
        # Check cache first
        cache_key = feature_set_name
        if not force_rebuild and cache_key in self._feature_cache:
            return self._feature_cache[cache_key]

        if self.raw_train is None or self.raw_test is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Start with raw data (excluding target column if present)
        train_df = self.raw_train.copy()
        test_df = self.raw_test.copy()

        # Remove target from train features if it somehow got in there
        if 'target' in train_df.columns:
            train_df = train_df.drop('target', axis=1)

        # Build features using the configuration
        if feature_set_name == "default":
            # Use the standard build_features function with config
            train_features = build_features(train_df, self.config.features)
            test_features = build_features(test_df, self.config.features)
        else:
            # For custom feature sets, we might want to use different logic
            # For now, fall back to default behavior
            train_features = build_features(train_df, self.config.features)
            test_features = build_features(test_df, self.config.features)

        # Store feature names
        feature_names = list(train_features.columns)

        # Cache the results
        self._feature_cache[cache_key] = (train_features, test_features, feature_names)

        return train_features, test_features, feature_names

    def get_train_val_split(self,
                           test_size: float = 0.2,
                           shuffle: bool = True,
                           random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Get a random train/validation split from the training data.
        Uses scikit-learn's train_test_split under the hood (appropriate for i.i.d. data).

        Args:
            test_size: Proportion of dataset to include in validation split
            shuffle: Whether to shuffle before splitting
            random_state: Random seed (uses config.cv.random_state if None)

        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        # Use config value if not provided
        if random_state is None:
            random_state = self.config.cv.random_state

        return self.get_train_test_split(
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state
        )

    def get_cv_generator(self,
                        n_splits: int = 5,
                        shuffle: bool = True,
                        random_state: Optional[int] = None):
        """
        Generate random train/validation splits for cross-validation.
        Appropriate for i.i.d. data (since dates are anonymized/shuffled).

        Yields:
            Tuples of (X_train, y_train, X_val, y_val) for each fold
        """
        if self.raw_train is None or self.y_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Use config value if not provided
        if random_state is None:
            random_state = self.config.cv.random_state

        # Create KFold splitter
        kf = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

        # Generate splits
        for train_idx, val_idx in kf.split(self.raw_train):
            # Extract the data for this split
            X_train = self.raw_train.iloc[train_idx]
            y_train = self.y_train.iloc[train_idx]
            X_val = self.raw_train.iloc[val_idx]
            y_val = self.y_train.iloc[val_idx]

            yield X_train, y_train, X_val, y_val

    def get_full_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get full training data and labels.

        Returns:
            Tuple of (X_train, y_train)
        """
        if self.raw_train is None or self.y_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return self.raw_train, self.y_train

    def get_test_data(self) -> pd.DataFrame:
        """
        Get test features (no labels available for test data).

        Returns:
            DataFrame containing test features
        """
        if self.raw_test is None:
            raise ValueError("Test data not loaded. Call load_data() first.")

        return self.raw_test

    def get_feature_names(self, feature_set_name: str = "default") -> List[str]:
        """
        Get the names of features for a given feature set.

        Args:
            feature_set_name: Name of feature set

        Returns:
            List of feature column names
        """
        _, _, feature_names = self.build_feature_set(feature_set_name)
        return feature_names

    def clear_cache(self) -> None:
        """Clear the feature cache to force recomputation."""
        self._feature_cache.clear()