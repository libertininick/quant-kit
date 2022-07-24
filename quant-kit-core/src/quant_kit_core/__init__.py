from os import path

from quant_kit_core.utils import get_timediff

ROOT_DIR = path.dirname(path.abspath(__file__))

__version__ = "1.0.0.dev0"

__all__ = [
    "ROOT_DIR",
    "get_timediff",
]
