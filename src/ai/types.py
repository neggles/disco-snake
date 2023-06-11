from collections import UserList
from typing import List, TypeAlias, Union

from disnake import DMChannel, GroupChannel, Member, TextChannel, Thread, User
from PIL.Image import Image

ListOfUsers = Union[List[Union[User, Member]], List[User], List[Member]]

MessageChannel = Union[TextChannel, DMChannel, GroupChannel, Thread]

ImageOrBytes: TypeAlias = Union[Image, bytes]


class RingBuffer(UserList):
    def __init__(self, initlist=None, size_max=32):
        self.max = size_max
        super().__init__(initlist=initlist)

    def append(self, x):
        """Append an element to the end of the buffer."""
        self.data.append(x)
        if len(self.data) == self.max:
            self.data.pop(0)

    def push(self, x):
        """Append an element to the end of the buffer."""
        self.append(x)
