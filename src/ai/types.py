from typing import List, TypeAlias, Union

from disnake import DMChannel, GroupChannel, Member, TextChannel, Thread, User
from PIL.Image import Image

ListOfUsers = Union[List[Union[User, Member]], List[User], List[Member]]

MessageChannel = Union[TextChannel, DMChannel, GroupChannel, Thread]

ImageOrBytes: TypeAlias = Union[Image, bytes]
