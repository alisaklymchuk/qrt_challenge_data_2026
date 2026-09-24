from sklearn.metrics import make_scorer
from . import MODEL_REGISTRY, add_model
from . import ridge  # Import ridge model to register it

def sign_acc(y_true, y_pred):
    return ((y_pred >= 0) == (y_true >= 0)).mean()

sign_acc_score = make_scorer(sign_acc)