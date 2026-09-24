from dataclasses import dataclass
import yaml

@dataclass
class DataConfig:
    train_path: str
    test_path: str
    target_path: str = None

@dataclass
class FeatureConfig:
    features: list[str]
    columns: list[str]
    horizons: dict[str, int | list[int]]

@dataclass
class ModelConfig:
    regressor: str
    params: dict

@dataclass
class CVConfig:
    n_splits: int
    shuffle: bool
    random_state: int

@dataclass
class ExperimentConfig:
    name: str
    seed: int

@dataclass
class Config:
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    cv: CVConfig
    experiment: ExperimentConfig

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

    data_config = DataConfig(**config_dict["data"])
    feature_config = FeatureConfig(**config_dict["features"])
    model_config = ModelConfig(**config_dict["model"])
    cv_config = CVConfig(**config_dict["cv"])
    experiment_config = ExperimentConfig(**config_dict["experiment"])

    return Config(
        data=data_config,
        features=feature_config,
        model=model_config,
        cv=cv_config,
        experiment=experiment_config,
    )