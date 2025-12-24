import json
import logging
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from disnake import (
    DMChannel,
    Emoji,
    GroupChannel,
    Guild,
    Interaction,
    Member,
    Message,
    Role,
    TextChannel,
    Thread,
    User,
)
from emoji import demojize, emojize

from ai.types import LruDict, MessageChannel
from disco_snake.bot import DiscoSnake

logger = logging.getLogger(__name__)

# timezone mappings for zones i've bothered to map
TZ_MAP = {
    "aest": ZoneInfo("Australia/Melbourne"),
    "jst": ZoneInfo("Asia/Tokyo"),
    "pst": ZoneInfo("America/Los_Angeles"),
    "est": ZoneInfo("America/New_York"),
}

# match multiple spaces
re_spaces = re.compile(r"\s\s+")

# capture discord mentions and emojis
re_mention = re.compile(r"<@(\d+)>", re.I)
re_emoji = re.compile(r"<(a)?(:[^:]+:)(\d+)>", re.I)
# capture mentions in bot responses
re_mention_resp = re.compile(r"\b(@\S+)\b", re.I)

# capture first word
re_firstword = re.compile(r"^\b(\S+)\b", re.M + re.I)


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


def shorten_spaces(text: str) -> str:
    """
    Remove extra spaces from a string (e.g. "  hello  world  " -> " hello world ")

    """
    return re_spaces.sub(" ", text)


# get the best possible author name for a message
def get_message_author_name(
    ctx: Message | Member | User | Interaction,
    suffix: str = "",
    first_word: bool = True,
) -> str:
    match ctx:
        case Member() | User():
            author = ctx
        case _:
            if hasattr(ctx, "author"):
                author = ctx.author
            elif hasattr(ctx, "user"):
                author = ctx.user
            else:
                raise TypeError("ctx must be Message, Member, User, or Interaction")

    if getattr(author, "nick", None):
        # use nickname if available
        author_name = author.nick
    elif getattr(author, "global_name", None):
        # use global name if available
        author_name = author.global_name
    elif getattr(author, "display_name", None):
        # use display name if available
        author_name = author.display_name
    elif getattr(author, "name", None):
        # use name if available
        author_name = author.name
    else:
        # fallback to str()
        author_name = str(author)

    # round-trip to ascii to remove any super weird characters
    author_name = str(author_name).encode("utf-8").decode("ascii", errors="ignore").strip()

    # optionally reduce to first word only
    if first_word:
        if match := re_firstword.search(author_name):
            author_name = match.group(1)

    # append suffix if any and return
    return author_name + suffix


class MentionMixin:
    """Mixin class for handling conversion between emojis/mentions and text
    Uses a fun LRU dict to store the last 100 messages' mentions/emojis
    for restoration, keyed by message ID.
    """

    bot: DiscoSnake
    _mention_cache: LruDict
    _emoji_cache: LruDict

    def __init__(self, mention_cache_size: int = 100, *args, **kwargs):
        self._mention_cache = LruDict(mention_cache_size)
        self._emoji_cache = LruDict(mention_cache_size)
        super().__init__(*args, **kwargs)

    def stringify_mentions_emoji(
        self,
        text: str,
        message: Message,
        unicode: bool = False,
    ) -> str:
        text, mentions = _stringify_mentions(self.bot, text, message.guild)
        self._mention_cache[message.id] = mentions
        text, emojis = _stringify_custom_emoji(text, unicode=unicode)
        self._emoji_cache[message.id] = emojis
        return text

    def restore_mentions_emoji(
        self,
        text: str,
        message: Message,
        unicode: bool = True,
    ) -> str:
        text = _restore_mentions(text, self.mention_cache(message.id))
        text = _restore_custom_emoji(text, self.emoji_cache(message.id), unicode=unicode)
        text = _map_response_mentions(text, message)
        return text

    def mention_cache(self, message_id: int) -> dict[str, str]:
        """Get the mention cache for a given message ID."""
        return self._mention_cache.get(message_id, {})

    def emoji_cache(self, message_id: int) -> dict[str, str]:
        """Get the emoji cache for a given message ID."""
        return self._emoji_cache.get(message_id, {})


def _stringify_mentions(bot: DiscoSnake, text: str, guild: Guild | None = None) -> tuple[str, dict[str, str]]:
    mentions = {}
    for mention in re_mention.finditer(text):
        user_mention = f"{mention.group(0)}"
        user_id = int(mention.group(1))
        user = None
        if guild is not None:
            user = guild.get_member(user_id)
        if user is None:
            user = bot.get_user(user_id)

        if user is not None:
            name_string = get_message_author_name(user)
        else:
            name_string = "@deleted-user"

        # store mention in dict
        mentions[name_string] = user_mention
        # replace mention with name string
        text = text.replace(user_mention, name_string)
    return text, mentions


def _restore_mentions(text: str, mentions: dict[str, str]) -> str:
    for name_string, user_mention in mentions.items():
        if name_string == "@deleted-user":
            continue  # skip deleted users
        # restore mention from LRU dict
        text = re.sub(r"@?" + re.escape(name_string) + r"\b", user_mention, text, flags=re.I)
    return text


def _stringify_custom_emoji(
    text: str,
    unicode: bool = True,
) -> tuple[str, dict[str, str]]:
    emojis = {}
    for match in re_emoji.finditer(text):
        anim_flag, emoji_name, emoji_id = match.groups()
        emojis[emoji_name] = anim_flag, emoji_id
        text = text.replace(match.group(), emoji_name)
    if unicode:
        text = demojize(text, language="alias")
    return text, emojis


def _restore_custom_emoji(
    text: str,
    emojis: dict[str, str],
    unicode: bool = True,
) -> str:
    for emoji_name, (anim_flag, emoji_id) in emojis.items():
        # restore emoji from LRU dict
        if anim_flag is None:
            anim_flag = ""
        text = text.replace(emoji_name, f"<{anim_flag}{emoji_name}{emoji_id}>")
    if unicode:
        text = emojize(text, language="alias")
    return text


def _map_response_mentions(response: str, ctx: Message) -> str:
    if not ctx.guild:
        return response

    for mention in re_mention_resp.finditer(response):
        user_mention = mention.group(0)
        user_name = user_mention.lstrip("@")

        mention_tag = None
        if user := ctx.guild.get_member_named(user_name):
            mention_tag = user.mention
        elif user := ctx.guild.get_member_named(user_name.lower()):
            mention_tag = user.mention
        elif user := ctx.guild.get_member_named(user_name.capitalize()):
            mention_tag = user.mention
        if mention_tag:
            response = response.replace(user_mention, mention_tag)
    return response


def get_lm_prompt_time(tz: str | ZoneInfo = ZoneInfo("Asia/Tokyo")) -> str:
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


def member_in_role(member: Member, role: Role | int) -> bool:
    """Returns True if the user has the role"""
    if isinstance(role, Role):
        role = role.id
    return role in [x.id for x in member.roles]


def member_in_any_role(member: Member, roles: list[Role | int]) -> bool:
    """Returns True if the user has any of the roles"""
    for role in roles:
        if member_in_role(member, role):
            return True
    return False


def json_dump(obj: Any, file: Path, indent: int = 4, sort_keys: bool = True) -> str:
    with file.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=sort_keys)
    return json.dumps(obj, indent=indent, sort_keys=sort_keys)


@lru_cache(maxsize=100, typed=True)
def _dedupe_custom_emoji(emojis: list[Emoji]) -> list[Emoji]:
    seen_names = set()
    deduped = []
    for emo in emojis:
        if not emo.is_usable():
            continue
        if emo.name not in seen_names:
            deduped.append(emo)
            seen_names.add(emo.name)
            continue
    return deduped


@lru_cache(maxsize=100, typed=True)
def get_usable_custom_emoji(context: Interaction | MessageChannel) -> list[Emoji]:
    """Get a list of available emojis for the message context"""
    bot: DiscoSnake = context.client  # type: ignore
    usable = []

    match context:
        case Interaction():
            # if we have permissions, use all emojis
            if context.app_permissions.external_emojis:
                usable = _dedupe_custom_emoji(bot.emojis)
        case TextChannel() | Thread():
            # if we have permissions, use all emojis
            if context.permissions_for(bot.user).external_emojis:
                usable = _dedupe_custom_emoji(bot.emojis)
        case DMChannel() | GroupChannel():
            # all emojis are available in DMs and group chats
            usable = _dedupe_custom_emoji(bot.emojis)
        case _:
            logger.error(f"Unknown context type for emoji retrieval: {context!r}")

    # fallback to guild emojis if we have none and we're in a guild context
    if not usable and context.guild:
        usable = _dedupe_custom_emoji(context.guild.emojis)

    return usable
