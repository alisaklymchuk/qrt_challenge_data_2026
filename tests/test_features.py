"""
Tests for feature builders.
"""
import pandas as pd
import numpy as np
from src.features import (
    FEATURE_REGISTRY,
    add_feature,
    get_base_feature_columns,
    apply_feature,
    build_features,
    create_benchmark_features,
    RETURN_COLS,
    VOL_COLS
)
from src.config import FeatureConfig


def test_feature_registry():
    """Test that the feature registry contains expected features."""
    assert 'momentum' in FEATURE_REGISTRY
    assert 'volatility' in FEATURE_REGISTRY
    assert 'zscore' in FEATURE_REGISTRY
    assert 'autocorr' in FEATURE_REGISTRY
    assert len(FEATURE_REGISTRY) >= 4


def test_get_base_feature_columns():
    """Test getting base feature columns."""
    returns_cols = get_base_feature_columns('returns')
    assert returns_cols == RETURN_COLS.copy()

    vol_cols = get_base_feature_columns('signed_volume')
    assert vol_cols == VOL_COLS.copy()

    turnover_cols = get_base_feature_columns('turnover')
    assert turnover_cols == ['MEDIAN_DAILY_TURNOVER']

    # Test invalid feature type
    empty_cols = get_base_feature_columns('invalid')
    assert empty_cols == []


def test_apply_feature():
    """Test applying a feature to data."""
    # Create sample data
    n_samples = 100
    data = {}
    for i in range(1, 21):
        data[f'RET_{i}'] = np.random.randn(n_samples)
        data[f'SIGNED_VOLUME_{i}'] = np.random.randn(n_samples)
    data['MEDIAN_DAILY_TURNOVER'] = np.random.rand(n_samples)
    df = pd.DataFrame(data)

    # Test momentum feature
    result = apply_feature(df, 'momentum', h=5, signals=['ret'])
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) == 1
    assert len(result) == n_samples
    # Check that the feature name is expected
    assert 'ret_momentum_h_5' in result.columns or 'momentum_h_5' in result.columns

    # Test volatility feature
    result = apply_feature(df, 'volatility', h=10, signals=['vol'])
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) == 1
    assert len(result) == n_samples


def test_build_features():
    """Test building features with a config."""
    # Create sample data
    n_samples = 50
    data = {}
    for i in range(1, 21):
        data[f'RET_{i}'] = np.random.randn(n_samples)
        data[f'SIGNED_VOLUME_{i}'] = np.random.randn(n_samples)
    data['MEDIAN_DAILY_TURNOVER'] = np.random.rand(n_samples)
    df = pd.DataFrame(data)

    # Create a mock config
    class MockConfig:
        features = ['momentum', 'volatility']
        horizons = {'momentum': [5, 10], 'volatility': [5]}

    config = MockConfig()
    result = build_features(df, config)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == n_samples
    # Should have 2 (momentum) * 2 (horizons) + 1 (volatility) * 1 (horizons) = 5 features
    assert len(result.columns) == 5


def test_create_benchmark_features():
    """Test creating benchmark features."""
    # Create sample data
    n_samples = 30
    data = {}
    for i in range(1, 21):
        data[f'RET_{i}'] = np.random.randn(n_samples)
        data[f'SIGNED_VOLUME_{i}'] = np.random.randn(n_samples)
    data['MEDIAN_DAILY_TURNOVER'] = np.random.rand(n_samples)
    data['TS'] = np.repeat(['A', 'B', 'C'], n_samples // 3)  # Group column
    df = pd.DataFrame(data)

    # Test with default parameters
    feature_names = create_benchmark_features(
        df,
        include_raw=True,
        include_derived=True,
        include_grouped=True
    )

    assert isinstance(feature_names, list)
    assert len(feature_names) > 0
    # Check that we have some expected feature types
    feature_string = ' '.join(feature_names)
    assert 'RET_' in feature_string or 'SIGNED_VOLUME_' in feature_string or 'MEDIAN_DAILY_TURNOVER' in feature_string


def test_add_feature_decorator():
    """Test that we can add a new feature using the decorator."""
    initial_count = len(FEATURE_REGISTRY)

    @add_feature("test_feature")
    def test_feature(df, cols, h=5):
        return pd.Series(np.ones(len(df)), index=df.index)

    assert 'test_feature' in FEATURE_REGISTRY
    assert len(FEATURE_REGISTRY) == initial_count + 1

    # Test that the feature works
    n_samples = 10
    data = {f'RET_{i}': np.random.randn(n_samples) for i in range(1, 6)}
    data['MEDIAN_DAILY_TURNOVER'] = np.random.rand(n_samples)
    df = pd.DataFrame(data)

    result = apply_feature(df, 'test_feature', h=3, signals=['ret'])
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) == 1
    assert len(result) == n_samples
    # All values should be 1.0 (from our test feature)
    assert (result.iloc[:, 0] == 1.0).all()


if __name__ == "__main__":
    test_feature_registry()
    test_get_base_feature_columns()
    test_apply_feature()
    test_build_features()
    test_create_benchmark_features()
    test_add_feature_decorator()
    print("All tests passed!")