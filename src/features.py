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

def apply_feature(df, feature_name, h=5):
    feature_fn = FEATURE_REGISTRY[feature_name]

    features = {}

    for signal, cols in SIGNAL_COLS.items():
        features[f"{signal}_{feature_name}_{h}"] = feature_fn(
            df,
            cols=cols,
            h=h,
        )

    return pd.DataFrame(features, index=df.index)

def build_features(df, config):
    features = []

    for feature_name in config.features:
        for feature, h in config.horizons.items():
            features.append(apply_feature(df, feature_name, h=h))

    return pd.concat(features, axis=1)