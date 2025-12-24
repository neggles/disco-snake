from collections.abc import Generator, MutableMapping
from datetime import datetime
from typing import Any, ClassVar

import sqlalchemy as sa

from db import SyncSession, SyncSessionType
from db.discord.user import BlacklistEntry


class Blacklist(MutableMapping):
    db_client: ClassVar[SyncSessionType] = SyncSession
    instance: ClassVar["Blacklist"] = None  # type: ignore

    def __new__(cls) -> "Blacklist":
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __getitem__(self, key: int) -> BlacklistEntry:
        with self.db_client.begin() as session:
            entry = session.get(BlacklistEntry, key, populate_existing=True)
        if entry is None:
            raise KeyError(key)
        return entry

    def __setitem__(self, key, value: BlacklistEntry | str):
        """Add a discord user to the blacklist.
        Key must be their user ID, value must be a BlacklistEntry or reason string.
        """
        if isinstance(value, str):
            value = BlacklistEntry(user_id=key, reason=value, timestamp=datetime.now())
        with self.db_client.begin() as session:
            session.merge(value)

    def __delitem__(self, key):
        with self.db_client.begin() as session:
            entry = session.get(BlacklistEntry, key)
            if entry is None:
                raise KeyError(key)
            session.execute(sa.delete(BlacklistEntry).where(BlacklistEntry.user_id == key))

    def __iter__(self) -> Generator[int, Any, None]:
        with self.db_client.begin() as session:
            for entry in session.query(BlacklistEntry):
                yield entry.user_id

    def __len__(self) -> int:
        with self.db_client.begin() as session:
            return session.scalar(sa.select(sa.func.count(BlacklistEntry.user_id))) or 0

    def add_id(self, id: int, reason: str = "No reason provided") -> None:
        self[id] = BlacklistEntry(user_id=id, reason=reason, timestamp=datetime.now())

    def remove_id(self, id: int) -> None:
        self.__delitem__(id)

    def all_ids(self) -> list[int]:
        return list(self.keys())
