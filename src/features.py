import pandas as pd


FEATURE_REGISTRY = {}

def add_feature(name):
    def decorator(func):
        FEATURE_REGISTRY[name] = func
        return func
    return decorator


RETURN_COLS = [f"RET_{i}" for i in range(1, 21)]
VOL_COLS = [f"SINGED_VOLUME{i}" for i in range(1, 21)]

@add_feature("momentum")
def momentum(df, cols, h=5):
    # approx as a sum since daily returns and short horizon
    mom = df[cols[:h]].mean(axis=1)
    return mom

@add_feature("volatility")
def volatility(df, cols, h=5):
    vol = df[cols[:h]].std(axis=1)
    return vol

@add_feature("zscore")
def zscore(df, cols, h=20):
    zscore = (df[cols[0]] - df[cols[:h]].mean(axis=1)) / (df[cols[:h]].std(axis=1) + 10**(-7))
    return zscore

@add_features("autocorr")
def autocorr(df, cols, h=20):
    return df