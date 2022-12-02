try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown (no version information available)"
    version_tuple = (0, 0, "unknown", "noinfo")

from pathlib import Path

from helpers.misc import get_package_root

LOGDIR_PATH = Path.cwd().joinpath("logs")
DATADIR_PATH = Path.cwd().joinpath("data")

CONFIG_PATH = DATADIR_PATH.joinpath("config.json")
USERDATA_PATH = DATADIR_PATH.joinpath("userdata.json")
BLACKLIST_PATH = DATADIR_PATH.joinpath("blacklist.json")

PACKAGE_ROOT = get_package_root()
COGDIR_PATH = PACKAGE_ROOT.joinpath("cogs")
EXTDIR_PATH = PACKAGE_ROOT.joinpath("extensions")

PACKAGE = __package__.replace("_", "-")
