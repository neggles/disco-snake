from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    pass


class NSFWLevel(int, Enum):
    DEFAULT = 0
    EXPLICIT = 1
    SAFE = 2
    AGE_RESTRICTED = 3
