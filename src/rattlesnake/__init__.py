try:
    from disco_snake._version import version as __version__
    from disco_snake._version import version_tuple
except ImportError:
    __version__ = "unknown (no version information available)"
    version_tuple = (0, 0, "unknown", "noinfo")

from disco_snake import DATADIR_PATH

RATTLESNAKE_DATA_DIR = DATADIR_PATH.joinpath("rattlesnake")
if not RATTLESNAKE_DATA_DIR.exists():
    RATTLESNAKE_DATA_DIR.mkdir(parents=True, exist_ok=True)
RATTLESNAKE_CONFIG_PATH = RATTLESNAKE_DATA_DIR.joinpath("config.json")
