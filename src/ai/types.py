from collections import OrderedDict
from datetime import datetime
from time import monotonic
from typing import Any, Hashable, TypeAlias, Union

import disnake
from cachetools import TTLCache
from disnake import DMChannel, GroupChannel, Member, Message, TextChannel, Thread, User
from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field

ListOfUsers: TypeAlias = Union[list[Union[User, Member]], list[User], list[Member]]

MessageChannel: TypeAlias = Union[TextChannel, DMChannel, GroupChannel, Thread]

ImageOrBytes: TypeAlias = Union[Image, bytes]


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


class AiMessageData(BaseModel):
    id: int = Field(...)
    timestamp: datetime = Field(...)
    guild_id: int | None = None
    guild: str = Field(...)
    author_id: int | None = None
    author: str = Field(...)
    channel_id: int | None = None
    channel: str = Field(...)
    author_name: str = Field(...)
    trigger: str = Field("Unknown")
    content: str = Field(...)
    conversation: list[str] = Field(...)

    model_config = ConfigDict(
        from_attributes=True,
    )


class AiResponseLog(BaseModel):
    message: AiMessageData = Field(...)
    gensettings: dict = Field(...)
    context: dict = Field(...)
    response_raw: str = Field(...)
    response: str = Field(...)

    model_config = ConfigDict(
        from_attributes=True,
    )


class AiResponse(BaseModel):
    ctx: Message = Field(...)
    content: str | None = None
    image: disnake.File | None = None
    file: disnake.File | None = None
    is_reply: bool = False

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    async def send(self):
        return await self.ctx.send(**self.send_kwargs())

    async def add_reaction(self, emoji: disnake.Emoji | disnake.PartialEmoji | str):
        return await self.ctx.add_reaction(emoji)

    def send_kwargs(self) -> dict[str, Any]:
        args: dict[str, Any] = {"content": self.content}

        if self.image and self.file:
            args["file"] = [self.image, self.file]
        elif self.image:
            args["file"] = self.image
        elif self.file:
            args["file"] = self.file

        if self.is_reply and isinstance(self.ctx, Message):
            args["reference"] = self.ctx.to_reference()

        return args
