from collections import OrderedDict
from datetime import datetime
from typing import List, Optional, TypeAlias, Union

from disnake import DMChannel, GroupChannel, Member, TextChannel, Thread, User
from PIL.Image import Image
from pydantic import BaseModel, Field

ListOfUsers: TypeAlias = Union[List[Union[User, Member]], List[User], List[Member]]

MessageChannel: TypeAlias = Union[TextChannel, DMChannel, GroupChannel, Thread]

ImageOrBytes: TypeAlias = Union[Image, bytes]


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
    guild_id: Optional[int] = Field(None)
    guild: str = Field(...)
    author_id: Optional[int] = Field(None)
    author: str = Field(...)
    channel_id: Optional[int] = Field(None)
    channel: str = Field(...)
    author_name: str = Field(...)
    trigger: str = Field("Unknown")
    content: str = Field(...)
    conversation: List[str] = Field(...)


class AiResponseLog(BaseModel):
    message: AiMessageData = Field(...)
    gensettings: dict = Field(...)
    context: dict = Field(...)
    response_raw: str = Field(...)
    response: str = Field(...)

    class Config:
        orm_mode = True
