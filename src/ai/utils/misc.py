import logging
import mimetypes
from base64 import b64encode
from collections import OrderedDict
from collections.abc import Hashable
from datetime import datetime
from time import monotonic
from zoneinfo import ZoneInfo

from cachetools import TTLCache

from ai.constants import TZ_MAP

logger = logging.getLogger(__name__)


class TimestampStore(TTLCache):
    def __init__(self, maxsize=128, ttl=90) -> None:
        super().__init__(maxsize, ttl, timer=monotonic)

    def refresh(self, key: Hashable) -> None:
        # update the timestamp for a key
        self[key] = True

    def active(self, key: Hashable) -> bool:
        return self.get(key, False)


class LruDict(OrderedDict):
    def __init__(self, max_size=100, other=(), /, **kwds):
        self.max_size = max_size
        super().__init__(other, **kwds)

    def __setitem__(self, key, value, *args, **kwargs):
        # Call the superclass method, then prune the dictionary if it's too big.
        super().__setitem__(key, value, *args, **kwargs)
        if len(self) > self.max_size:
            self.popitem(last=False)


def cleanup_thoughts(thoughts: list[str]) -> list[str]:
    """Cleans up a list of thought strings by stripping leading and trailing whitespace and removing <think> tags."""
    thoughts = [x.strip() for x in thoughts]

    # find the index of the first non-empty line
    start_line = 0
    for idx, line in enumerate(thoughts):
        if line != "":
            start_line = idx
            break

    # find the index of the last non-empty line
    n_lines = len(thoughts) - 1
    for idx in range(len(thoughts) - 1, -1, -1):
        if thoughts[idx] != "":
            n_lines = idx
            break

    # extract relevant lines and remove <think> tags
    cleaned = []
    for line in thoughts[start_line : n_lines + 1]:
        line = line.strip()
        if line.startswith("<think>"):
            if line := line.replace("<think>", "").strip():
                cleaned.append(line)
        elif line.endswith("</think>"):
            if line := line.replace("</think>", "").strip():
                cleaned.append(line)
        else:
            # keep empty lines in the middle
            cleaned.append(line)
    return cleaned


def get_date_suffixed(day: int, with_num: bool = True) -> str:
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = ["st", "nd", "rd"][day % 10 - 1]
    return f"{day}{suffix}" if with_num else suffix


def get_prompt_datetime(tz: str | ZoneInfo = ZoneInfo("Asia/Tokyo"), with_date: bool = False) -> str:
    if not isinstance(tz, ZoneInfo):
        if tz not in TZ_MAP.keys():
            raise ValueError(f"Unmapped timezone: {tz}")
        tz = TZ_MAP[tz]
    now = datetime.now(tz=tz)
    suffix_date = get_date_suffixed(now.day)

    fmt_string = f"%-I:%M%p on %A, {suffix_date} %B %Y" if with_date else "%-I:%M %p"
    return datetime.now(tz=tz).strftime(fmt_string)


def any_in_text(strings: list[str], text: str) -> bool:
    """Returns True if any of the strings are in the text"""
    return any([s in text for s in strings])


def data_uri_from_bytes(
    data: bytes,
    filename: str | None,
    fallback_mime: str = "application/octet-stream",
) -> str:
    """Convert bytes to a data URI string."""
    if not data:
        raise ValueError("data cannot be empty")
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")

    mime, _ = mimetypes.guess_type(filename or "")
    mime = mime if mime else fallback_mime
    b64 = b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"
