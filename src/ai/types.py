from typing import List, Union

from disnake import DMChannel, GroupChannel, Member, TextChannel, Thread, User

ListOfUsers = Union[List[Union[User, Member]], List[User], List[Member]]
MessageChannel = Union[TextChannel, DMChannel, GroupChannel, Thread]
