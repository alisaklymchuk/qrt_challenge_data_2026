"""Models package."""

MODEL_REGISTRY = {}


def add_model(name):
    def decorator(fn):
        MODEL_REGISTRY[name] = fn
        return fn
    return decorator