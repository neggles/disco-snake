from collections import OrderedDict
from typing import List, TypeAlias, Union

from disnake import DMChannel, GroupChannel, Member, TextChannel, Thread, User
from PIL.Image import Image

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
