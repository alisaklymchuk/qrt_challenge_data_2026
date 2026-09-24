from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import pandas as pd
from . import MODEL_REGISTRY, add_model


def sign_acc(y_true, y_pred):
    return ((y_pred >= 0) == (y_true >= 0)).mean()


sign_acc_score = make_scorer(sign_acc)


@add_model("ridge")
def build_ridge(X_train, y_train, **params):

    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(**params))
    ])
    cv_params = {"ridge__alpha": [0.01, 0.1, 1, 10, 100]}
    cv = GridSearchCV(ridge, cv_params, scoring=sign_acc_score)
    cv.fit(X_train, y_train)
    return cv.best_estimator_