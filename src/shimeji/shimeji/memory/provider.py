from typing import Optional
from datetime import datetime, timezone

from shimeji.memory import Memory

EPOCH = 1621123998


def snowflake():
    return int((datetime.now(timezone.utc).timestamp() - EPOCH) * 10**5)


def to_snowflake(timestamp: int):
    return int((timestamp - EPOCH) * 10**5)


class MemoryStore:
    """
    Base class for MemoryStores.
    Represents the interface that all MemoryStores must implement.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a MemoryStore.
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
