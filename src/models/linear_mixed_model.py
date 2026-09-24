import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import pandas as pd
from . import MODEL_REGISTRY, add_model

@add_model("linear_mixed_model")
def build_linear_mixed_model(X_train, y_train, **params):
    features = [col for col in X_train.columns if col != "GROUP"]
    X = sm.add_constant(X_train[features])

    model = sm.MixedLM(
        endog=y_train,
        exog=X,
        groups=X_train["GROUP"]
    )

    result = model.fit()
    return result

