import logging
import re
from pathlib import Path


def get_package_root() -> Path:
    return Path(__file__).parent.parent


def parse_log_level(level: str | int) -> int:
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


re_filename = re.compile(r"[^0-9a-zA-Z+-._ ]")


def filename_filter(s: str) -> str:
    return re_filename.sub("", s)
