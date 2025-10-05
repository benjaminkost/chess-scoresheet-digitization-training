from pathlib import Path

def get_root_dir_path() -> Path:
    return Path(__file__).parent.parent.parent.parent