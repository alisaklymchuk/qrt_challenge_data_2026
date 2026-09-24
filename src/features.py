import pandas as pd


FEATURE_REGISTRY = {}


def add_feature(name):
    def decorator(func):
        FEATURE_REGISTRY[name] = func
        return func

    return decorator


RETURN_COLS = [f"RET_{i}" for i in range(1, 21)]
VOL_COLS = [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]

SIGNAL_COLS = {
    "ret": RETURN_COLS,
    "vol": VOL_COLS,
}


@add_feature("momentum")
def momentum(df, cols, h=5):
    # Approx. cumulative return as a sum for short daily-return horizons
    return df[cols[:h]].mean(axis=1)


@add_feature("volatility")
def volatility(df, cols, h=5):
    return df[cols[:h]].std(axis=1)


@add_feature("zscore")
def zscore(df, cols, h=20):
    x = df[cols[:h]]

    return (
        (x.iloc[:, 0] - x.mean(axis=1))
        / (x.std(axis=1) + 1e-7)
    )


@add_feature("autocorr")
def autocorr(df, cols, h=20):
    x = df[cols[:h]].to_numpy()

    x1 = x[:, :-1]
    x2 = x[:, 1:]

    x1_mean = x1.mean(axis=1, keepdims=True)
    x2_mean = x2.mean(axis=1, keepdims=True)

    numerator = ((x1 - x1_mean) * (x2 - x2_mean)).sum(axis=1)

    denominator = (
        ((x1 - x1_mean) ** 2).sum(axis=1) ** 0.5
        * ((x2 - x2_mean) ** 2).sum(axis=1) ** 0.5
    )

    return pd.Series(
        numerator / (denominator + 1e-7),
        index=df.index,
    )


def get_base_feature_columns(feature_type):
    """
    Get column names for base feature types.

    Args:
        feature_type: One of 'returns', 'signed_volume', 'turnover'

    Returns:
        List of column names for the specified feature type
    """
    if feature_type == 'returns':
        return RETURN_COLS.copy()
    elif feature_type == 'signed_volume':
        return VOL_COLS.copy()
    elif feature_type == 'turnover':
        return ['MEDIAN_DAILY_TURNOVER']
    else:
        return []


def apply_feature(df, feature_name, h=5, signals=None, columns=None, feature_name_prefix=None):
    """
    Apply a feature transformation to specified columns or signal groups.

    Args:
        df: DataFrame to apply feature to
        feature_name: Name of feature function from FEATURE_REGISTRY
        h: Horizon parameter (passed to feature function)
        signals: List of signal keys to apply to (None = all from SIGNAL_COLS)
        columns: Explicit column list to use instead of SIGNAL_COLS
        feature_name_prefix: Prefix for output feature names (defaults to feature_name)

    Returns:
        DataFrame with feature(s) as columns
    """
    feature_fn = FEATURE_REGISTRY[feature_name]
    prefix = feature_name_prefix or feature_name
    features = {}

    # Determine which columns to use
    if columns is not None:
        # Use explicit column list
        cols_to_use = columns
        signal_name = "custom"
    elif signals is not None:
        # Use specified signal groups
        cols_to_use = []
        for signal in signals:
            if signal in SIGNAL_COLS:
                cols_to_use.extend(SIGNAL_COLS[signal])
        # Create reasonable name from signals
        signal_name = "_".join(signals) if signals else "custom"
    else:
        # Default: apply to all signals
        cols_to_use = []
        signal_names = []
        for signal, cols in SIGNAL_COLS.items():
            cols_to_use.extend(cols)
            signal_names.append(signal)
        signal_name = "_".join(signal_names)

    if cols_to_use:
        result = feature_fn(df, cols=cols_to_use, h=h)
        # Create feature name
        if columns is not None:
            feat_name = f"{prefix}_h_{h}" if h is not None else prefix
        elif signals is not None and len(signals) == 1:
            feat_name = f"{signals[0]}_{prefix}_h_{h}" if h is not None else f"{signals[0]}_{prefix}"
        else:
            feat_name = f"{signal_name}_{prefix}_h_{h}" if h is not None else f"{signal_name}_{prefix}"

        features[feat_name] = result

    return pd.DataFrame(features, index=df.index)


def build_features(df, config):
    """
    Build features from dataframe using config.

    Enhanced to handle horizons as lists and more flexible feature specification.

    Args:
        df: Input dataframe
        config: Config object with .features and .horizons attributes
                .features: list of feature names to apply
                .horizons: dict mapping feature name to horizon(s) (int or list of ints)

    Returns:
        DataFrame with all features as columns
    """
    features = []

    for feature_name in config.features:
        horizons = config.horizons[feature_name]
        # Handle both single int and list of ints
        if not isinstance(horizons, list):
            horizons = [horizons]

        for h in horizons:
            features.append(apply_feature(df, feature_name, h=h))
    if hasattr(config, "columns"):
        features.append(df[config.columns])

    return pd.concat(features, axis=1)


def create_benchmark_features(df, include_raw=True, include_derived=True,
                            include_grouped=True, return_windows=[3, 5, 10, 15, 20],
                            std_windows=[20], volume_windows=[5]):
    """
    Create features following the pattern from benchmark_submission.ipynb.

    This function makes it easy to replicate the feature engineering from the notebook.

    Args:
        df: Input dataframe with RETURN_COLS, VOL_COLS, MEDIAN_DAILY_TURNOVER, and 'TS' column
        include_raw: Whether to include raw base features
        include_derived: Whether to include derived features (rolling mean/std)
        include_grouped: Whether to include group-aggregated features (by TS)
        return_windows: List of window sizes for return-based features
        std_windows: List of window sizes for return-based std features
        volume_windows: List of window sizes for signed volume-based features

    Returns:
        List of feature column names that were created
    """
    feature_names = []

    # Get base feature columns
    RET_features = get_base_feature_columns('returns')
    VOL_features = get_base_feature_columns('signed_volume')
    TURNOVER_features = get_base_feature_columns('turnover')

    # 1. Base features (raw)
    if include_raw:
        # Add raw return features
        for col in RET_features:
            df[col] = df[col]  # Ensure column exists
            feature_names.append(col)

        # Add raw signed volume features
        for col in VOL_features:
            df[col] = df[col]  # Ensure column exists
            feature_names.append(col)

        # Add turnover feature
        for col in TURNOVER_features:
            df[col] = df[col]  # Ensure column exists
            feature_names.append(col)

    # 2. Derived features (rolling statistics on returns)
    if include_derived:
        # Mean features: AVERAGE_PERF_{window}
        for window in return_windows:
            feat_name = f'AVERAGE_PERF_{window}'
            df[feat_name] = df[RET_features[:window]].mean(axis=1)
            feature_names.append(feat_name)

        # Std features: STD_PERF_{window}
        for window in std_windows:
            feat_name = f'STD_PERF_{window}'
            df[feat_name] = df[RET_features[:window]].std(axis=1)
            feature_names.append(feat_name)

        # Volume mean features: SIGNED_VOLUME_MEAN_{window}
        for window in volume_windows:
            feat_name = f'SIGNED_VOLUME_MEAN_{window}'
            df[feat_name] = df[VOL_features[:window]].mean(axis=1)
            feature_names.append(feat_name)

    # 3. Grouped features (group mean by TS)
    if include_grouped and 'TS' in df.columns:
        # Group mean of return-based features
        for window in return_windows:
            base_feat = f'AVERAGE_PERF_{window}'
            if base_feat in df.columns:
                group_feat = f'ALLOCATIONS_{base_feat}'
                df[group_feat] = df.groupby('TS')[base_feat].transform('mean')
                feature_names.append(group_feat)

        # Group std of return-based features
        for window in std_windows:
            base_feat = f'STD_PERF_{window}'
            if base_feat in df.columns:
                group_feat = f'ALLOCATIONS_{base_feat}'
                df[group_feat] = df.groupby('TS')[base_feat].transform('mean')
                feature_names.append(group_feat)

        # Group mean of volume-based features
        for window in volume_windows:
            base_feat = f'SIGNED_VOLUME_MEAN_{window}'
            if base_feat in df.columns:
                group_feat = f'ALLOCATIONS_{base_feat}'
                df[group_feat] = df.groupby('TS')[base_feat].transform('mean')
                feature_names.append(group_feat)

    return feature_names