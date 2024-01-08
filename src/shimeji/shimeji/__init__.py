__title__ = "shimeji"
__version__ = "0.1.1"
__author__ = "hitomi-team"
__license__ = "GPLv2 License"
__copyright__ = "Copyright 2022 hitomi-team"

__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from shimeji.shimeji import ChatBot

from shimeji import memory, model_provider, postprocessor, preprocessor, tokenizers, util

__all__ = [
    "ChatBot",
    "memory",
    "model_provider",
    "postprocessor",
    "preprocessor",
    "tokenizers",
    "util",
]
