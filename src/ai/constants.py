from zoneinfo import ZoneInfo

from humanize import naturalsize

from disco_snake import (
    LOG_FORMAT,
    LOGDIR_PATH,
    PACKAGE_ROOT,
    per_config_name,
)

# folder paths
AI_DATA_DIR = PACKAGE_ROOT.parent.joinpath("data", "ai")
AI_IMAGES_DIR = AI_DATA_DIR.joinpath(per_config_name("images"))
AI_LOG_DIR = LOGDIR_PATH

# logging format
AI_LOG_FORMAT = LOG_FORMAT

# timezone mappings for zones i've bothered to map
TZ_MAP = {
    "aest": ZoneInfo("Australia/Melbourne"),
    "jst": ZoneInfo("Asia/Tokyo"),
    "pst": ZoneInfo("America/Los_Angeles"),
    "est": ZoneInfo("America/New_York"),
}

# maximums for image handling
MAX_IMAGE_WIDTH = 1280
MAX_IMAGE_HEIGHT = 1280
MAX_IMAGE_EDGE = max(MAX_IMAGE_WIDTH, MAX_IMAGE_HEIGHT)
MAX_IMAGE_BYTES = 20 * (2**20)  # 20 MB, we will downscale larger images
MAX_IMAGES_PER_MESSAGE = 4  # be reasonable

# Human-readable strings for logging
MAX_IMAGE_BYTES_STR = naturalsize(MAX_IMAGE_BYTES, binary=True)

# api client settings
MAX_API_RETRIES = 2

# exported names
__all__ = [
    "AI_DATA_DIR",
    "AI_IMAGES_DIR",
    "AI_LOG_DIR",
    "AI_LOG_FORMAT",
    "MAX_IMAGE_WIDTH",
    "MAX_IMAGE_HEIGHT",
    "MAX_IMAGE_EDGE",
    "MAX_IMAGE_BYTES",
    "MAX_IMAGES_PER_MESSAGE",
    "MAX_IMAGE_BYTES_STR",
    "MAX_API_RETRIES",
]
