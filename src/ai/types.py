from collections import OrderedDict
from datetime import datetime
from time import monotonic
from typing import Any, TypeAlias
from collections.abc import Hashable

import disnake
from cachetools import TTLCache
from disnake import DMChannel, GroupChannel, Member, Message, TextChannel, Thread, User
from PIL.Image import Image
from pydantic import BaseModel, ConfigDict, Field

ListOfUsers: TypeAlias = list[User | Member] | list[User] | list[Member]

MessageChannel: TypeAlias = TextChannel | DMChannel | GroupChannel | Thread

ImageOrBytes: TypeAlias = Image | bytes


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
        extra="allow",
    )


class AiResponseLog(BaseModel):
    message: AiMessageData
    gensettings: dict
    context: dict
    response_raw: str
    response: str

    model_config = ConfigDict(
        from_attributes=True,
        extra="allow",
    )


class AiResponse(BaseModel):
    ctx: Message = Field(...)
    content: str = ""
    image: disnake.File | None = None
    file: disnake.File | None = None
    is_reply: bool = False

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    async def send(self):
        if self.content is None and self.image is None and self.file is None:
            raise ValueError("Cannot send an response with no content, image, or file.")
        if self.ctx is None:
            raise ValueError("Cannot send an response with no context (ctx).")

        if hasattr(self.ctx, "send"):
            return await self.ctx.send(**self.send_kwargs())
        elif hasattr(self.ctx, "channel") and hasattr(self.ctx.channel, "send"):
            return await self.ctx.channel.send(**self.send_kwargs())
        else:
            raise ValueError(f"Context {self.ctx} does not support sending messages.")

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


class ModelParameterInfo(BaseModel):
    max_seq_len: int = Field(...)
    cache_size: int = -1
    cache_mode: str | None = None
    rope_scale: float | None = None
    rope_alpha: float | None = None
    max_batch_size: int = -1
    chunk_size: int = -1
    prompt_template: str | None = None
    prompt_template_content: str | None = Field(None, repr=False)
    use_vision: bool = False
    draft: dict[str, Any] | None = None

    model_config = ConfigDict(
        from_attributes=True,
        extra="allow",
    )


class ModelInfo(BaseModel):
    id: str = Field(...)
    object: str = Field(...)
    created: int = Field(...)
    owned_by: str = Field(...)
    logging: dict[str, bool] | None = None
    parameters: ModelParameterInfo = Field(...)

    model_config = ConfigDict(
        from_attributes=True,
        extra="allow",
    )
