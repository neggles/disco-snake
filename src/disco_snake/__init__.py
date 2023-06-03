try:
    from ._version import (
        version as __version__,
        version_tuple,
    )
except ImportError:
    __version__ = "unknown (no version information available)"
    version_tuple = (0, 0, "unknown", "noinfo")

from argparse import ArgumentParser
from functools import lru_cache
from pathlib import Path
from sys import argv
from typing import Optional

from helpers.misc import get_package_root

LOG_FORMAT = "%(color)s[%(levelname)1.1s %(asctime)s][%(name)s][%(module)s:%(funcName)s:%(lineno)d]%(end_color)s %(message)s"

PACKAGE = __package__.replace("_", "-")
PACKAGE_ROOT = get_package_root()

COGDIR_PATH = PACKAGE_ROOT.joinpath("cogs")
LOGDIR_PATH = PACKAGE_ROOT.parent.joinpath("logs")
DATADIR_PATH = PACKAGE_ROOT.parent.joinpath("data")
DEF_CONFIG_PATH = DATADIR_PATH.joinpath("config.json")


@lru_cache(maxsize=2)
def get_suffix(path: bool = False) -> Optional[str]:
    """oh this? this is a war crime.

    This needs to be callable from the CLI, and it needs to be callable *before* any of the config
    loading happens. So we can't just use the config to get the suffix, because the config hasn't
    been loaded yet. So we have to do this.

    I'm sorry. I'm so sorry.
    """
    # ok so we make our heinous argument parser
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", default=DEF_CONFIG_PATH)
    # we parse the args and throw away the leftovers
    args, _ = parser.parse_known_args(args=argv[1:])
    # we get our config path from the args
    config_path = Path(args.config)
    # we get the name of the config file without the extension, and remove the "config-" part
    suffix = config_path.stem.replace("config", "").lstrip("-")
    # we'll just return None if it's the default/empty suffix and we don't want the path
    if path is False:
        return f"{suffix}" if suffix != "" else None
    # otherwise we make the daemonocle prefix path and return it
    return config_path.parent.joinpath(f"daemon/{suffix}")
    # then either way we go turn ourselves in to the police


def get_suffix_name(name: str, extension: Optional[str] = None) -> str:
    """Append the config suffix to a name, if it exists."""
    if extension is not None:
        extension = f".{extension}" if not extension.startswith(".") else extension
    else:
        extension = ""
    suffix = get_suffix()
    return f"{name}{extension}" if suffix is None else f"{name}-{suffix}{extension}"


if get_suffix() is not None:
    LOGDIR_PATH = LOGDIR_PATH.joinpath(get_suffix())
    DATADIR_PATH = DATADIR_PATH.joinpath(get_suffix())

BLACKLIST_PATH = DATADIR_PATH.joinpath("blacklist.json")
MISCDATA_PATH = DATADIR_PATH.joinpath("misc")
