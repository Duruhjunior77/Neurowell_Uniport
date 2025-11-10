import os, yaml

def project_root():
    # repo root = parent of src/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def load_cfg(path: str = None):
    """Load config.yaml from explicit path or <repo_root>/config.yaml"""
    if path is None:
        path = os.path.join(project_root(), "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def makedirs(path: str):
    os.makedirs(path, exist_ok=True)
