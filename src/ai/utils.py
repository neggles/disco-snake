import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple, Union
from zoneinfo import ZoneInfo

from disnake import Guild, Member, Message, Role

from ai.types import LruDict
from disco_snake.bot import DiscoSnake

logger = logging.getLogger(__name__)

# for stripping extra spaces from ai responses
re_spaces = re.compile(r"\s+")

# capture mentions and emojis
re_mention = re.compile(r"<@(\d+)>", re.I)
re_emoji = re.compile(r"<(a)?(:[^:]+:)(\d+)>", re.I)

# capture mentions in bot responses
re_mention_resp = re.compile(r"(@\w+)\b", re.I)


def cleanup_thoughts(thoughts: list[str]) -> list[str]:
    """Cleans up a list of thought strings by stripping leading and trailing whitespace and removing <think> tags."""
    thoughts = [x.strip() for x in thoughts]
    n_lines = len(thoughts)
    cleaned = []

    start_line = 0
    for idx, line in enumerate(thoughts):
        if line != "":
            start_line = idx
            break
    last_text_idx = n_lines
    for idx in range(n_lines, -1, -1):
        if thoughts[idx] != "":
            last_text_idx = idx
            break

    for line in thoughts[start_line : last_text_idx + 1]:
        line = line.strip()
        if line.startswith("<think>") or line.endswith("</think>"):
            if line := line.replace("<think>", "", 1).replace("</think>", "", 1).strip():
                cleaned.append(line)
            continue
        else:
            # keep empty lines in the middle
            cleaned.append(line)
    return cleaned


def shorten_spaces(text: str) -> str:
    """
    Remove extra spaces from a string (e.g. "  hello  world  " -> " hello world ")

    """
    return re_spaces.sub(" ", text)


class MentionMixin:
    """Mixin class for handling conversion between emojis/mentions and text
    Uses a fun LRU dict to store the last 100 messages' mentions/emojis
    for restoration, keyed by message ID.
    """

    bot: DiscoSnake
    _mention_cache: LruDict
    _emoji_cache: LruDict

    def __init__(self, max_size: int = 100, *args, **kwargs):
        self._mention_cache = LruDict(max_size)
        self._emoji_cache = LruDict(max_size)

    def stringify_mentions_emoji(
        self,
        text: str,
        message: Message,
    ) -> str:
        text, mentions = _stringify_mentions(self.bot, text, message.guild)
        text, emojis = _stringify_emoji(text)
        self._mention_cache[message.id] = mentions
        self._emoji_cache[message.id] = emojis
        return text

    def restore_mentions_emoji(self, text: str, message: Message) -> str:
        text = _restore_mentions(text, self._mention_cache[message.id])
        text = _restore_emoji(text, self._emoji_cache[message.id])
        text = _map_response_mentions(text, message)
        return text


def _stringify_mentions(
    bot: DiscoSnake, text: str, guild: Optional[Guild] = None
) -> Tuple[str, dict[str, str]]:
    mentions = {}
    for mention in re_mention.finditer(text):
        user_mention = f"{mention.group(0)}"
        user_id = int(mention.group(1))
        user = None
        if guild is not None:
            user = guild.get_member(user_id)
        if user is None:
            user = bot.get_user(user_id)

        name_string = "@deleted-user" if user is None else f"@{user.display_name}"
        # store mention in dict
        mentions[name_string] = user_mention
        # replace mention with display name
        text = text.replace(user_mention, name_string)
    return text, mentions


def _restore_mentions(text: str, mentions: dict[str, str]) -> str:
    for name_string, user_mention in mentions.items():
        if name_string == "@deleted-user":
            continue  # skip deleted users
        # restore mention from LRU dict
        text = re.sub(r"@?" + re.escape(name_string) + r"\b", user_mention, text, flags=re.I)
    return text


def _stringify_emoji(text: str) -> Tuple[str, dict[str, str]]:
    emojis = {}
    for match in re_emoji.finditer(text):
        anim_flag, emoji_name, emoji_id = match.groups()
        emojis[emoji_name] = anim_flag, emoji_id
        text = text.replace(match.group(), emoji_name)
    return text, emojis


def _restore_emoji(text: str, emojis: dict[str, str]) -> str:
    for emoji_name, (anim_flag, emoji_id) in emojis.items():
        # restore emoji from LRU dict
        if anim_flag is None:
            anim_flag = ""
        text = text.replace(emoji_name, f"<{anim_flag}{emoji_name}{emoji_id}>")
    return text


def _map_response_mentions(response: str, ctx: Message) -> str:
    for mention in re_mention_resp.finditer(response):
        user_mention = mention.group(0)
        user_name = user_mention.lstrip("@")
        if ctx.guild is None:
            continue

        user = ctx.guild.get_member_named(user_name)
        if user is None:
            user = ctx.guild.get_member_named(user_name.lower())
        if user is None:
            continue

        response = response.replace(user_mention, user.mention)
    return response


# timezone maps for below
TZ_MAP = {
    "aest": ZoneInfo("Australia/Melbourne"),
    "jst": ZoneInfo("Asia/Tokyo"),
    "pst": ZoneInfo("America/Los_Angeles"),
    "est": ZoneInfo("America/New_York"),
}


def get_lm_prompt_time(tz: Union[str, ZoneInfo] = ZoneInfo("Asia/Tokyo")) -> str:
    if not isinstance(tz, ZoneInfo):
        if tz not in TZ_MAP.keys():
            raise ValueError(f"Unmapped timezone: {tz}")
        tz = TZ_MAP[tz]
    return datetime.now(tz=tz).strftime("%-I:%M %p")


def get_date_suffixed(day: int, with_num: bool = True) -> str:
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = ["st", "nd", "rd"][day % 10 - 1]
    return f"{day}{suffix}" if with_num else suffix


def get_prompt_datetime(tz: Union[str, ZoneInfo] = ZoneInfo("Asia/Tokyo"), with_date: bool = False) -> str:
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


def member_in_role(member: Member, role: Union[Role, int]) -> bool:
    """Returns True if the user has the role"""
    if isinstance(role, Role):
        role = role.id
    return role in [x.id for x in member.roles]


def member_in_any_role(member: Member, roles: list[Union[Role, int]]) -> bool:
    """Returns True if the user has any of the roles"""
    for role in roles:
        if member_in_role(member, role):
            return True
    return False


def json_dump(obj: Any, file: Path, indent: int = 4, sort_keys: bool = True) -> str:
    with file.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=sort_keys)
    return json.dumps(obj, indent=indent, sort_keys=sort_keys)
