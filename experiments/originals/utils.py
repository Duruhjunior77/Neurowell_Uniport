
# neurowell/src/utils.py
import os, yaml

def load_cfg(path: str = None):
    if path is None:
        # Automatically locate the config.yaml one level above neurowell/
        path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        path = os.path.abspath(path)
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
