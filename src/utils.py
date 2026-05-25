import os

def ensure_dirs():
    folders = [
        "data/processed",
        "outputs",
        "metrics",
        "predictions",
        "models/bert"
    ]

    for folder in folders:
        os.makedirs(
            folder,
            exist_ok=True
        )