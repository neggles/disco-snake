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
DEF_DATA_PATH = DATADIR_PATH
DEF_DAEMON_PATH = DATADIR_PATH.joinpath("daemon")


@lru_cache(maxsize=2)
def config_suffix(daemon_path: bool = False) -> Optional[str | Path]:
    """oh this? this is a war crime. don't look at it. please. i'm begging you.

    Retrieves the config suffix from the CLI args. This needs to be callable from the CLI,
    and it needs to be callable *before* any of the config loading happens, because the config
    file contains the suffix. So we have to do this. I'm sorry. I'm so sorry.

    passing daemon_path=True will return the path to the daemonocle prefix directory instead of the string suffix.
    """
    # ok so we make our heinous argument parser
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", default=None, required=False, help="config name to use")
    # we parse the args, throwing away the ones we don't know about, and get our config name
    args, _ = parser.parse_known_args(args=argv[1:])
    config_name = args.config

    # if we're using the default config, we can return now
    if config_name is None:
        return DEF_DAEMON_PATH if daemon_path is True else None

    # otherwise we need to check if the config file exists
    config_path = DEF_DATA_PATH.joinpath(f"config-{config_name}.json")
    if config_path.exists() is False:
        raise FileNotFoundError(f"Config file {config_path} does not exist!")

    # and return either the name or the daemonocle prefix path
    return DEF_DAEMON_PATH.joinpath(f"{config_name}") if daemon_path is True else config_name
    # then either way we go turn ourselves in to the police


def per_config_name(name: str, extension: Optional[str] = None) -> str:
    """Append the config suffix to a name, if it exists."""
    if extension is None and "." in name:
        # if we don't have an extension, but the name has one, split it
        name, extension = name.rsplit(".", 1)
    # get our config suffix and append it to the name if it exists
    suffix = config_suffix()
    name = f"{name}-{suffix}" if suffix is not None else name

    # and return the name with the extension if provided
    return f"{name}.{extension.lstrip('.')}" if extension is not None else name


if config_suffix() is not None:
    LOGDIR_PATH = LOGDIR_PATH.joinpath(config_suffix())
    DATADIR_PATH = DATADIR_PATH.joinpath(config_suffix())

BLACKLIST_PATH = DATADIR_PATH.joinpath("blacklist.json")
MISCDATA_PATH = DATADIR_PATH.joinpath("misc")
