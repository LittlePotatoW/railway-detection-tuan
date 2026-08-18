import yaml
from pathlib import Path

with open(Path(__file__).parent / 'config.yaml', 'r', encoding='utf-8') as file:
    Config = yaml.safe_load(file)

