from sklearn.metrics import make_scorer

def sign_acc(y_true, y_pred):
    return ((y_pred >= 0) == (y_true >= 0)).mean()

sign_acc_score = make_scorer(sign_acc)

MODEL_REGISTRY = {}

def add_model(name):
    def decorator(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator