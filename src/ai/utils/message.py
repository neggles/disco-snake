import logging
import re

from disnake import (
    Guild,
    Interaction,
    Member,
    Message,
    User,
)
from emoji import demojize, emojize

from disco_snake.bot import DiscoSnake

from .misc import LruDict
from .regex import re_emoji, re_firstword, re_mention, re_mention_resp

logger = logging.getLogger(__name__)


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
