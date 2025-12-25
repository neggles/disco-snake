import logging
import re
from time import perf_counter

logger = logging.getLogger(__name__)

### Compiled regex patterns
# TODO: is there much benefit to compiling all of these?
#       Could they maybe be lazy-compiled without impacting the interface?
logger.debug("Compiling regex patterns")
_compile_start = perf_counter()

# match multiple spaces
re_spaces = re.compile(r"\s\s+")

# capture first word
re_firstword = re.compile(r"^\b(\S+)\b", re.M + re.I)

# capture non-word characters
re_nonword = re.compile(r"\W+", re.M + re.I)

# capture content in between angle brackets
re_angle_bracket = re.compile(r"\<(.*)\>", re.M)

# capture user and bot tokens, this is probably unnecessary these days
re_user_token = re.compile(r"(<USER>|<user>|{{user}})", re.I)
re_bot_token = re.compile(r"(<bot>|{{bot}}|<char>|{{char}}|<assistant>|{{assistant}})", re.I)

# unescape markdown characters
re_unescape_md = re.compile(r"\\([*_~`\"])", re.M)

# match lines that look like "Name: " at the start of a line
re_linebreak_name = re.compile(r"[\n\r]*(\S+):\s", re.I + re.M)
# match lines that start with *Expression*
re_start_expression = re.compile(r"^\s*[(\*]\w+[)\*]\s*", re.I + re.M)
# find sentences that start with uppercase letters
re_upper_first = re.compile(r"^([A-Z]\s?[^A-Z])")
# find URLs
re_detect_url = re.compile(
    r"[(http(s)?):\/\/(www\.)?a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)",
    re.M + re.I,
)

# find consecutive newlines (at least 3) with optional spaces in between (blank lines)
re_consecutive_newline = re.compile(r"(\n[\s\n]+\n\s*)", re.M + re.I)

# capture discord mentions and emojis
re_mention = re.compile(r"<@(\d+)>", re.I)
re_emoji = re.compile(r"<(a)?(:[^:]+:)(\d+)>", re.I)

# capture mentions in bot responses
re_mention_resp = re.compile(r"\b(@\S+)\b", re.I)

_compile_end = perf_counter()
logger.debug(f"Compiled regex patterns in {_compile_end - _compile_start:.4f} seconds")
### End regex patterns


### Common functions using regex patterns
def re_match_lower(match: re.Match):
    """function for re.sub() to convert the first match to lowercase"""
    return match.group(1).lower()
