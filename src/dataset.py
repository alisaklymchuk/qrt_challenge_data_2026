"""
Dataset class for handling financial time-series data and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from typing import Tuple, List, Optional
from .features import get_base_feature_columns, apply_feature, build_features, create_benchmark_features
from .config import Config


class QuantDataset:
    """
    A dataset class for handling financial time-series data with feature engineering
    capabilities and time-series aware train/validation splits.

    This class encapsulates:
    - Loading of separate X_train.csv, X_test.csv, y_train.csv files
    - Feature engineering using configurable transformations
    - Time-series aware train/validation split generation
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

        Matches the pattern used in benchmark_submission.ipynb
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

    def get_timestamp_column(self) -> pd.Series:
        """
        Get the timestamp column from training data.

        Returns:
            pandas Series containing the TS values
        """
        if self.raw_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.raw_train['TS']

    def get_unique_dates(self) -> np.ndarray:
        """
        Get unique dates/timestamps from training data.

        Returns:
            numpy array of unique timestamp values
        """
        return self.get_timestamp_column().unique()

    def get_date_splits(self,
                       n_splits: int = 5,
                       shuffle: bool = True,
                       random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate time-series aware train/validation splits based on unique dates.

        This replicates the approach used in benchmark_submission.ipynb:
        1. Get unique dates from training data
        2. Apply KFold splitting on the dates
        3. For each fold, get the row indices belonging to train/validation dates

        Args:
            n_splits: Number of splits for cross-validation
            shuffle: Whether to shuffle dates before splitting
            random_state: Random seed for reproducibility

        Returns:
            List of tuples (train_indices, val_indices) for each fold
        """
        if self.raw_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Get unique dates from training data
        dates = self.get_unique_dates()

        # Apply KFold on dates (exactly like benchmark_submission.ipynb)
        kf = KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

        splits = []
        for train_date_idx, val_date_idx in kf.split(dates):
            train_dates = dates[train_date_idx]
            val_dates = dates[val_date_idx]

            # Get row indices for these dates
            train_mask = self.raw_train['TS'].isin(train_dates)
            val_mask = self.raw_train['TS'].isin(val_dates)

            train_indices = self.raw_train[train_mask].index.values
            val_indices = self.raw_train[val_mask].index.values

            splits.append((train_indices, val_indices))

        return splits

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
                           fold_idx: int = 0,
                           feature_set_name: str = "default",
                           n_splits: Optional[int] = None,
                           shuffle: Optional[bool] = None,
                           random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Get training and validation data for a specific fold.

        Args:
            fold_idx: Which fold to return (0-indexed)
            feature_set_name: Name of feature set to use
            n_splits: Number of splits (uses config.cv.n_splits if None)
            shuffle: Whether to shuffle (uses config.cv.shuffle if None)
            random_state: Random seed (uses config.cv.random_state if None)

        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        # Use config values if not provided
        if n_splits is None:
            n_splits = self.config.cv.n_splits
        if shuffle is None:
            shuffle = self.config.cv.shuffle
        if random_state is None:
            random_state = self.config.cv.random_state

        # Get the date splits
        splits = self.get_date_splits(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

        if fold_idx >= len(splits):
            raise ValueError(f"fold_idx {fold_idx} is out of range for {len(splits)} splits")

        train_idx, val_idx = splits[fold_idx]

        # Get raw data for this split
        train_raw = self.raw_train.loc[train_idx]
        val_raw = self.raw_train.loc[val_idx]
        y_train_split = self.y_train.loc[train_idx]
        y_val_split = self.y_train.loc[val_idx]

        # Build features for this split
        # Temporarily replace the raw data in the dataset to build features correctly
        original_train = self.raw_train
        self.raw_train = train_raw
        try:
            X_train, _, _ = self.build_feature_set(feature_set_name, force_rebuild=True)
        finally:
            self.raw_train = original_train

        # Build validation features using the same approach
        self.raw_train = val_raw
        try:
            X_val, _, _ = self.build_feature_set(feature_set_name, force_rebuild=True)
        finally:
            self.raw_train = original_train

        return X_train, y_train_split, X_val, y_val_split

    def get_test_data(self, feature_set_name: str = "default") -> pd.DataFrame:
        """
        Get test features (no labels available for test data).

        Args:
            feature_set_name: Name of feature set to use

        Returns:
            DataFrame containing test features
        """
        _, test_features, _ = self.build_feature_set(feature_set_name)
        return test_features

    def get_full_train_data(self, feature_set_name: str = "default") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get full training data and labels.

        Args:
            feature_set_name: Name of feature set to use

        Returns:
            Tuple of (X_train, y_train)
        """
        if self.raw_train is None or self.y_train is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        X_train, _, _ = self.build_feature_set(feature_set_name)
        return X_train, self.y_train

    def get_cv_generator(self,
                        feature_set_name: str = "default",
                        n_splits: Optional[int] = None,
                        shuffle: Optional[bool] = None,
                        random_state: Optional[int] = None):
        """
        Generate train/validation splits for cross-validation.

        Yields:
            Tuples of (X_train, y_train, X_val, y_val) for each fold
        """
        if n_splits is None:
            n_splits = self.config.cv.n_splits
        if shuffle is None:
            shuffle = self.config.cv.shuffle
        if random_state is None:
            random_state = self.config.cv.random_state

        splits = self.get_date_splits(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            # Access train_idx and val_idx to avoid lint warnings (they're used implicitly)
            _ = train_idx
            _ = val_idx
            yield self.get_train_val_split(
                fold_idx=fold_idx,
                feature_set_name=feature_set_name,
                n_splits=1,  # We already have the specific split
                shuffle=False,  # Already shuffled in get_date_splits
                random_state=random_state
            )

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