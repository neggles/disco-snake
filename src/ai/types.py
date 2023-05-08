import re
from dataclasses import dataclass
from typing import List, Union

from disnake import Member, User, TextChannel, DMChannel, GroupChannel, Thread


ListOfUsers = Union[List[Union[User, Member]], List[User], List[Member]]
MessageChannel = Union[TextChannel, DMChannel, GroupChannel, Thread]
