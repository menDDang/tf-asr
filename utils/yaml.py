import yaml


def load_yaml(file_path: str) -> dict:
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.load(file, Loader=yaml.SafeLoader)
        