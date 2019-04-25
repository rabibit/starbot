import os


def mkdirs(path: str):
    # This require Python 3.4.1 or later
    os.makedirs(path, exist_ok=True)
