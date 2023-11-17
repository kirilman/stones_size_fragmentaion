import yaml


def load_config(config_path: str) -> None:
    config = yaml.safe_load(open(config_path))
    return config
