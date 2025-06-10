"""Top-level package for extractstoryboards."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """xuhuan2048"""
__email__ = "457938831@qq.com"
__version__ = "1.0.0"

from .src.extractstoryboards.nodes import NODE_CLASS_MAPPINGS
from .src.extractstoryboards.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
