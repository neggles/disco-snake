import json
import logging
import re
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import Levenshtein as lev
from disnake import Emoji, Guild, Member, Message, Role

from ai.types import ListOfUsers
from disco_snake.bot import DiscoSnake

logger = logging.getLogger(__name__)

# for stripping extra spaces from ai responses
re_spaces = re.compile(r"\s+")

# capture mentions and emojis
re_mention = re.compile(r"<@(\d+)>", re.I)
re_emoji = re.compile(r"<:([^:]+):(\d+)>", re.I)


def shorten_spaces(text: str) -> str:
    """
    Remove extra spaces from a string (e.g. "  hello  world  " -> " hello world ")

    """
    return re_spaces.sub(" ", text)


def anti_spam(messages: Union[List[Message], Message], threshold=0.8) -> Tuple[List[Message], int]:
    # Put messages in a list if only one message is passed
    if not isinstance(messages, list):
        messages = [messages]

    # Remove messages that are too similar to an earlier message
    spam = set()
    for msgnum, msg in enumerate(messages):
        for idx in range(msgnum + 1, len(messages)):
            if "<ctxbreak>" in msg.content:
                continue
            if SequenceMatcher(None, msg.content, messages[idx].content).ratio() > threshold:
                spam.add(messages[idx].id)

    # Return filtered messages and number of removed messages
    return [msg for msg in messages if msg.id not in spam], len(spam)


def standardize_punctuation(text: str) -> str:
    text = text.replace("’", "'")
    text = text.replace("`", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return text


def fix_trailing_quotes(text: str) -> str:
    num_quotes = text.count('"')
    if num_quotes % 2 == 0:
        return text
    else:
        return text + '"'


def fix_trailing_asterisk(text: str) -> str:
    num_asterisks = text.count("*")
    if num_asterisks % 2 == 0:
        return text
    else:
        return text + "*"


def cut_trailing_sentence(text: str) -> str:
    text = standardize_punctuation(text)
    last_punc = max(
        text.rfind("."),
        text.rfind("!"),
        text.rfind("?"),
        text.rfind("*"),
    )
    if last_punc <= 0:
        last_punc = len(text) - 1

    et_token = text.find("<")
    if et_token > 0:
        last_punc = min(last_punc, et_token - 1)

    text = text[: last_punc + 1]
    text = fix_trailing_quotes(text)
    text = fix_trailing_asterisk(text)
    return text


def get_full_class_name(obj: Any) -> str:
    module = obj.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return obj.__class__.__name__
    return module + "." + obj.__class__.__name__


def get_role_by_name(name: str, guild: Optional[Guild]) -> Optional[Role]:
    if guild is None:
        return None
    for role in guild.roles:
        if role.name == name:
            return role
    return None


def restore_mentions(text: str, users: ListOfUsers) -> str:
    # sort users from largest username to smallest
    users.sort(key=lambda user: len(user.name), reverse=True)

    for user in users:
        text = text.replace(f"@{user.name}", f"<@{user.id}>")

    return shorten_spaces(text)


def get_item(obj, key):
    if key in obj:
        return obj[key]
    else:
        return None


def get_levenshtein_distance(src: str, tgt: str, **kwargs) -> int:
    return lev.distance(src, tgt, **kwargs)


def get_subst_cost(c1: str, c2: str) -> int:
    if len(c1) > 1 or len(c2) > 1:
        raise ValueError("Expected single characters")
    return 0 if c1 == c2 else 1 if c1.isalpha() == c2.isalpha() else 2


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


def any_in_text(strings: list[str], text: str) -> bool:
    """Returns True if any of the strings are in the text"""
    return any([s in text for s in strings])


def member_in_role(member: Member, role: Union[Role, int]) -> bool:
    """Returns True if the user has the role"""
    if isinstance(role, Role):
        role = role.id
    return role in [x.id for x in member.roles]


def member_in_any_role(member: Member, roles: List[Union[Role, int]]) -> bool:
    """Returns True if the user has any of the roles"""
    for role in roles:
        if member_in_role(member, role):
            return True
    return False


def json_dump(obj: Any, file: Path, indent: int = 4, sort_keys: bool = True) -> str:
    with file.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=sort_keys)
    return json.dumps(obj, indent=indent, sort_keys=sort_keys)
