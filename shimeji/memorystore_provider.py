import time
from pydantic import BaseModel
from difflib import SequenceMatcher
from typing import Optional

sukima_epoch = 1621123998


def snowflake():
    return int((time.time() - sukima_epoch) * 10**5)


def to_snowflake(timestamp):
    return int((timestamp - sukima_epoch) * 10**5)


class Memory(BaseModel):
    """
    A BaseModel for representing a memory.
    """

    created_at: int
    author_id: int
    author: str
    text: str
    encoding_model: str
    encoding: str


class MemoryStoreProvider:
    """
    This is an abstract class that represents the interface that all MemoryStoreProviders would implement.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a MemoryStoreProvider.
        """
        self.kwargs = kwargs

    async def count(self) -> int:
        """
        Return the number of memories in the MemoryStore.

        :return: The number of memories in the MemoryStore.
        :rtype: int
        """
        raise NotImplementedError("count() is not implemented")

    async def filter(self, exclude_duplicates: bool, exclude_duplicates_ratio: float = None):
        """
        Returns a list of memories from the MemoryStore.

        :param exclude_duplicates: Exclude duplicate memories from the search.
        :type exclude_duplicates: bool
        :param exclude_duplicates_ratio: Exclude duplicates based upon Sequence Matching techniques. This can be set to None if that is not desired.
        :type exclude_duplicates_ratio: float
        """
        raise NotImplementedError("filter() is not implemented")

    async def create(self, memory: Memory) -> Optional[Memory]:
        """
        Add a memory directly to the MemoryStore.

        :param memory: The memory to add to the MemoryStore.
        :type Memory: Memory
        """
        raise NotImplementedError("set() is not implemented")

    async def delete(self, memory: Memory) -> Optional[Memory]:
        """
        Delete a memory from the MemoryStore.

        :param memory: The memory to delete from the MemoryStore.
        :type Memory: Memory
        """
        raise NotImplementedError("delete() is not implemented")

    async def add(self, author_id: int, author: str, text: str, encoding_model: str, encoding: str) -> Memory:
        """
        Create and add a memory to the MemoryStore.

        :param author_id: The ID of the author of the memory.
        :type author_id: int
        :param author: The name of the author of the memory.
        :type author: str
        :param text: The text of the memory.
        :type text: str
        :param encoding_model: The name of the encoding model used to encode the memory.
        :type encoding_model: str
        :param encoding: The encoding of the memory.
        :type encoding: str
        :rtype: Memory
        """
        raise NotImplementedError("add() is not implemented")

    async def check_duplicates(self, text: str, duplicate_ratio: float) -> bool:
        """
        Check if a memory is a duplicate.

        :param text: The text of the memory.
        :type text: str
        :param duplicate_ratio: The ratio of the text that must match to be considered a duplicate.
        :type duplicate_ratio: float
        :rtype: bool
        """
        raise NotImplementedError("check_duplicates() is not implemented")


from typing import AsyncIterator, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from shimeji.sqlcrud import memory as memorydb


class PostgreSQLMemoryStore(MemoryStoreProvider):
    """
    A MemoryStoreProvider using PostgreSQL.
    """

    def __init__(self, database_uri: str, **kwargs):
        """
        Initialize a PostgreSQLMemoryStore.

        :param database_uri: The URI of the database to connect to.
        """
        self.kwargs = kwargs

        self.engine = create_async_engine(database_uri, pool_pre_ping=True)
        self.async_session = sessionmaker(self.engine, expire_on_commit=False, class_=AsyncSession)

        self.model: str = self.kwargs.get("model", None)
        self.model_layer: int = self.kwargs.get("model_layer", -1)
        self.short_term_amount: int = self.kwargs.get("short_term_amount", 10)
        self.long_term_amount: int = self.kwargs.get("long_term_amount", 10)

    async def count(self) -> int:
        """
        Return the number of memories in the PostgreSQLMemoryStore.

        :return: The number of memories in the PostgreSQLMemoryStore.
        :rtype: int
        """
        async with self.async_session() as session, session.begin():
            return await memorydb.count(session=session)

    async def get_session(self) -> AsyncIterator[AsyncSession]:
        """
        Get a session from the database.

        :return: A session from the database.
        :rtype: AsyncSession
        """
        async with self.async_session() as session:
            try:
                yield session
            except Exception as e:
                raise e
            finally:
                await session.close()

    async def get(self, created_after: int = None) -> List[Memory]:
        """
        Get a list of memories created after a certain amount of time from the database.

        :param created_after:
        :type MemorySQL
        :rtype: List[Memory]
        """
        if created_after is None:
            created_after = 0

        async with self.async_session() as session, session.begin():
            return await memorydb.get_after_id(session=session, created_at=created_after)

    async def create(self, memory: Memory) -> Optional[Memory]:
        """
        Add a memory to the PostgreSQLMemoryStore.

        :param memory: The memory to add to the PostgreSQLMemoryStore.
        :type Memory: Memory
        :rtype: Memory
        """
        async with self.async_session() as session, session.begin():
            return await memorydb.create(
                session=session,
                created_at=memory.created_at,
                author_id=memory.author_id,
                author=memory.author,
                text=memory.text,
                encoding_model=memory.encoding_model,
                encoding=memory.encoding,
            )

    async def delete(self, memory: Memory) -> Optional[Memory]:
        """
        Delete a memory from the PostgreSQLMemoryStore. Only deletes based on the memory's created_at.

        :param memory: The memory to delete from the PostgreSQLMemoryStore.
        :type Memory: Memory
        :rtype: Memory
        """
        async with self.async_session() as session, session.begin():
            return await memorydb.delete(session=session, author_id=memory.author_id)

    async def add(self, author_id: int, author: str, text: str, encoding_model: str, encoding: str) -> Memory:
        """
        Create and add a memory to the MemoryStore.

        :param author_id: The ID of the author of the memory.
        :type author_id: int
        :param author: The name of the author of the memory.
        :type author: str
        :param text: The text of the memory.
        :type text: str
        :param encoding_model: The name of the encoding model used to encode the memory.
        :type encoding_model: str
        :param encoding: The encoding of the memory.
        :type encoding: str
        :rtype: Memory
        """

        memory = Memory(
            created_at=snowflake(),
            author_id=author_id,
            author=author,
            text=text,
            encoding_model=encoding_model,
            encoding=encoding,
        )

        return await self.create(memory=memory)

    async def filter(
        self, exclude_duplicates: bool = False, exclude_duplicates_ratio: float = 0.8
    ) -> List[Memory]:
        """
        Filter the memories based on a search string.

        :param exclude_duplicates: Exclude duplicates based upon Sequence Matching techniques. This can be set to None if that is not desired.
        :type exclude_duplicates: bool
        :param exclude_duplicates_ratio: Exclude duplicates based upon Sequence Matching techniques. This can be set to None if that is not desired.
        :type exclude_duplicates_ratio: float
        :rtype: List[Memory]
        """

        memories = await self.get(created_after=0)
        if exclude_duplicates:
            for m in memories:
                for m2 in memories:
                    if m.text == m2.text:
                        memories.remove(m2)
                    if exclude_duplicates_ratio:
                        if SequenceMatcher(None, m.text, m2.text).ratio() > exclude_duplicates_ratio:
                            memories.remove(m2)

        return memories

    async def check_duplicates(self, text: str, duplicate_ratio: float) -> bool:
        """
        Check if a memory is a duplicate.

        :param text: The text of the memory.
        :type text: str
        :param duplicate_ratio: The ratio of the text that must match to be considered a duplicate.
        :type duplicate_ratio: float
        :rtype: bool
        """
        memories = await self.get(created_after=0)

        for m in memories:
            if m.text == text or SequenceMatcher(None, m.text, text).ratio() > duplicate_ratio:
                return True

        return False
