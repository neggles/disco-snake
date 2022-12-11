import logging
from pathlib import Path
from typing import Union


def get_package_root() -> Path:
    return Path(__file__).parent.parent


def parse_log_level(level: Union[str, int]) -> int:
    if isinstance(level, str):
        level = level.lower()
        if level.startswith(("deb", "ver")):
            return logging.DEBUG
        elif level.startswith(("inf", "def")):
            return logging.INFO
        elif level.startswith("war"):
            return logging.WARNING
        elif level.startswith("err"):
            return logging.ERROR
        elif level.startswith("crit"):
            return logging.CRITICAL
        elif level.startswith("fatal"):
            return logging.FATAL
        elif level.startswith(("none", "not")):
            return logging.NOTSET
        else:
            raise ValueError(f"Unable to parse loglevel: {level}")
    else:
        return level


FILENAME_CHARS = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-._ ")


def filename_filter(s: str) -> str:
    return "".join(filter(lambda x: x in FILENAME_CHARS, s))
