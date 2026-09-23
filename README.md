# qrt_challenge_data_2026
My go at QRT's 2026 Challenge Data task: Asset Allocation Performance Forecasting. https://challengedata.ens.fr/participants/challenges/167/

## Project structure

```text
qrt_challenge/
│
├── run.py                         # Main entry point: loads a config and orchestrates the experiment
│
├── configs/
│   ├── default.yaml              # Default experiment configuration
│   ├── ridge_features_v1.yaml    # Example Ridge experiment
│   └── xgb_features_v1.yaml      # Example XGBoost experiment
│
├── results/
│   ├── ridge_features_v1/        # All outputs from one experiment
│   │   ├── config.yaml            # Exact configuration used for the run
│   │   ├── metrics.json           # Overall evaluation metrics
│   │   ├── fold_metrics.csv       # Metrics for each CV fold
│   │   ├── oof_predictions.parquet # Out-of-fold predictions
│   │   └── test_predictions.csv   # Predictions on the challenge test set
│   │
│   └── ...
│
├── src/
│   ├── __init__.py               # Makes src a Python package
│   │
│   ├── config.py                 # Dataclasses and YAML → Config loading
│   │
│   ├── data.py                   # Loading and basic preparation of train/test data
│   │
│   ├── features.py               # Feature definitions and feature registry
│   │
│   ├── models.py                 # Model construction / model registry
│   │
│   ├── evaluation.py             # CV evaluation, metrics, predictions and evaluation outputs
│   │
│   └── utils.py                  # Small shared utilities used across the pipeline
│
├── notebooks/
│   └── ...                       # Exploratory analysis and visualisation only
│
├── README.md
└── requirements.txt
```

### Responsibilities

The main principle is:

```text
configs/  → WHAT experiment to run
src/      → HOW the experiment works
run.py    → orchestrates the experiment
results/  → WHAT happened
```

### `run.py`

`run.py` is the single entry point for executing an experiment.

It should not contain the implementation of individual modelling or evaluation steps. Its job is to connect the components together:

```text
load config
    ↓
load data
    ↓
build features
    ↓
create model
    ↓
run cross-validation
    ↓
evaluate predictions
    ↓
save results
```

An experiment is executed with:

```bash
python run.py configs/ridge_features_v1.yaml
```

The config path is supplied as a command-line argument, so the same `run.py` can execute every experiment.

---

### `src/config.py`

Contains the configuration dataclasses and the function that converts YAML into a `Config` object.

The configuration is represented by:

```python
@dataclass
class Config:
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    cv: CVConfig
    experiment: ExperimentConfig
```

This means the rest of the code can use:

```python
config.features.enabled
config.model.regressor
config.model.params
config.cv.n_splits
config.experiment.name
```

rather than repeatedly accessing raw YAML dictionaries.

---

### `src/data.py`

Responsible for loading the dataset and performing basic data preparation.

For example:

```python
train = load_data(config.data.train_path)
test = load_data(config.data.test_path)
```

This module should deal with the dataset itself, but not with modelling decisions.

---

### `src/features.py`

Contains the feature engineering logic.

Features are registered centrally and can then be selected through the configuration.

For example:

```yaml
features:
  enabled:
    - momentum
    - volatility
    - zscore
    - autocorr
```

The feature module knows **how** to calculate these features.

The config decides **which** ones to use.

---

### `src/models.py`

Responsible for constructing the model specified in the config.

For example:

```yaml
model:
  regressor: ridge
  params:
    alpha: 1.0
```

The model module turns this into the appropriate estimator.

Later the same interface can support:

```text
ridge
elastic_net
lightgbm
xgboost
mlp
cnn
transformer
...
```

without changing `run.py`.

---

### `src/evaluation.py`

Contains the evaluation logic.

This is where predictions become metrics.

For example:

```text
predictions
    ↓
convert predictions to signs
    ↓
compare against TARGET
    ↓
calculate accuracy
    ↓
calculate fold metrics
    ↓
save metrics / predictions
```

It can also handle the cross-validation loop if we decide that CV is part of evaluation.

The important distinction is:

> `run.py` decides **when and in what order** things happen.
> `evaluation.py` defines **how predictions are evaluated**.

---

### `src/utils.py`

Small general-purpose helpers that don't naturally belong elsewhere.

Keep this relatively small. If a function becomes substantial enough to have a clear responsibility, it should probably get its own module instead.

---

## Configuration

Each experiment has its own YAML configuration.

Example:

```yaml
data:
  train_path: data/train.csv
  test_path: data/test.csv
  target: TARGET

features:
  enabled:
    - momentum
    - volatility
    - zscore
    - autocorr

  horizons:
    momentum: 5
    volatility: 20
    zscore: 20
    autocorr: 20

model:
  regressor: ridge
  params:
    alpha: 1.0

cv:
  n_splits: 5
  shuffle: true
  random_state: 42

experiment:
  name: ridge_features_v1
  seed: 42
```

### `data`

Defines where the data comes from and which column is the target.

```yaml
data:
  train_path: data/train.csv
  test_path: data/test.csv
  target: TARGET
```

### `features`

Defines which engineered features should be calculated and their parameters.

```yaml
features:
  enabled:
    - momentum
    - volatility
    - zscore
    - autocorr

  horizons:
    momentum: 5
    volatility: 20
    zscore: 20
    autocorr: 20
```

This allows us to change the feature set without modifying Python code.

For example, a raw-data experiment could eventually have:

```yaml
features:
  enabled: []
```

while another experiment could use:

```yaml
features:
  enabled:
    - momentum
    - volatility
    - zscore
    - autocorr
    - trend
    - drawdown
```

### `model`

Defines which model to use and its hyperparameters.

```yaml
model:
  regressor: ridge
  params:
    alpha: 1.0
```

The same experiment runner can therefore execute different models simply by changing the config.

### `cv`

Defines the cross-validation procedure.

```yaml
cv:
  n_splits: 5
  shuffle: true
  random_state: 42
```

The CV configuration belongs to the experiment because it is part of the experimental setup and should be recorded alongside the model and features.

### `experiment`

Contains metadata specific to the experiment.

```yaml
experiment:
  name: ridge_features_v1
  seed: 42
```

The experiment name determines where the results are stored:

```text
results/ridge_features_v1/
```

The seed ensures that stochastic parts of the experiment can be reproduced.

---

## Experiment workflow

Running:

```bash
python run.py configs/ridge_features_v1.yaml
```

will conceptually produce:

```text
configs/ridge_features_v1.yaml
             │
             ▼
          run.py
             │
     ┌───────┼────────┐
     ▼       ▼        ▼
   data   features   model
     │       │        │
     └───────┼────────┘
             ▼
       cross-validation
             │
             ▼
        predictions
             │
             ▼
      evaluation.py
             │
       ┌─────┴─────┐
       ▼           ▼
    metrics      predictions
       │           │
       └─────┬─────┘
             ▼
   results/ridge_features_v1/
```

The configuration used for the experiment is copied into the result directory:

```text
results/
└── ridge_features_v1/
    ├── config.yaml
    ├── metrics.json
    ├── fold_metrics.csv
    ├── oof_predictions.parquet
    └── test_predictions.csv
```

This makes each result directory self-contained: the predictions and metrics can always be traced back to the exact configuration that produced them.
