import re
from difflib import SequenceMatcher
from typing import Any, List, Optional, Tuple, Union

from disnake import Emoji, Guild, Message, Role, Member, User

from ai.types import ListOfUsers

# for stripping extra spaces from ai responses
RE_SPACES = re.compile(r"\s+")


def shorten_spaces(text: str) -> str:
    """
    Remove extra spaces from a string (e.g. "  hello  world  " -> " hello world ")

    """
    return RE_SPACES.sub(" ", text)


def anti_spam(messages: Union[List[Message], Message], threshold=0.8) -> Tuple[List[Message], int]:
    # Put messages in a list if only one message is passed
    if not isinstance(messages, list):
        messages = [messages]

    # Remove messages that are too similar to an earlier message
    spam = set()
    for msgnum, msg in enumerate(messages):
        for idx in range(msgnum + 1, len(messages)):
            if SequenceMatcher(None, msg.content, messages[idx].content).ratio() > threshold:
                spam.add(messages[idx].id)

    # Return filtered messages and number of removed messages
    return [msg for msg in messages if msg.id not in spam], len(spam)


def restore_mentions_emotes(text: str, users: ListOfUsers, emojis: List[Emoji]) -> str:
    # sort users from largest username to smallest
    users.sort(key=lambda user: len(user.name), reverse=True)

    for user in users:
        text = text.replace(f"@{user.name}", f"<@{user.id}>")

    for emoji in emojis:
        text = text.replace(f":{emoji.name}:", f"<:{emoji.name}:{emoji.id}>")

    return shorten_spaces(text)


def convert_mentions_emotes(text: str, users: ListOfUsers, emojis: List[Emoji]) -> str:
    for user in users:
        text = text.replace(f"<@{user.id}>", f"@{user.name}")

    for emoji in emojis:
        text = text.replace(f"<:{emoji.name}:{emoji.id}>", f":{emoji.name}:")

    return text


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
    if text[0] == "*":
        num_asterisks = text.count("*")
        if num_asterisks % 2 == 0:
            return text
        else:
            return text + "*"
    else:
        return text


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
