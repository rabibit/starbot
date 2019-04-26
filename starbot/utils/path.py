import os


def mkdirs(path: str):
    path = os.path.abspath(path)
    # This require Python 3.4.1 or later
    os.makedirs(path, exist_ok=True)
