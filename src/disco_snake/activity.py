import datetime
from typing import Any, Dict, Optional

from disnake.activity import ActivityTimestamps, ActivityType, BaseActivity


class ListeningActivity(BaseActivity):
    """A slimmed down version of :class:`Activity` that represents a Discord listening activity.

    This is typically displayed via **Listening to** on the official Discord client.

    .. container:: operations

        .. describe:: x == y

            Checks if two listens are equal.

        .. describe:: x != y

            Checks if two listens are not equal.

        .. describe:: hash(x)

            Returns the song's hash.

        .. describe:: str(x)

            Returns the song's name.

    Parameters
    ----------
    name: :class:`str`
        The song's name.

    Attributes
    ----------
    name: :class:`str`
        The song's name.
    """

    __slots__ = ("name", "_end", "_start")

    def __init__(self, name: str, **extra):
        super().__init__(**extra)
        self.name: str = name

        try:
            timestamps: ActivityTimestamps = extra["timestamps"]
        except KeyError:
            self._start = 0
            self._end = 0
        else:
            self._start = timestamps.get("start", 0)
            self._end = timestamps.get("end", 0)

    @property
    def type(self) -> ActivityType:
        """:class:`ActivityType`: Returns the song's type. This is for compatibility with :class:`Activity`.

        It always returns :attr:`ActivityType.playing`.
        """
        return ActivityType.listening

    @property
    def start(self) -> Optional[datetime.datetime]:
        """Optional[:class:`datetime.datetime`]: When the user started listening to this audio in UTC, if applicable."""
        if self._start:
            return datetime.datetime.fromtimestamp(self._start / 1000, tz=datetime.timezone.utc)
        return None

    @property
    def end(self) -> Optional[datetime.datetime]:
        """Optional[:class:`datetime.datetime`]: When the user will stop listening to this audio in UTC, if applicable."""
        if self._end:
            return datetime.datetime.fromtimestamp(self._end / 1000, tz=datetime.timezone.utc)
        return None

    def __str__(self) -> str:
        return str(self.name)

    def __repr__(self) -> str:
        return f"<Song name={self.name!r}>"

    def to_dict(self) -> Dict[str, Any]:
        timestamps: Dict[str, Any] = {}
        if self._start:
            timestamps["start"] = self._start

        if self._end:
            timestamps["end"] = self._end

        # fmt: off
        return {
            'type': ActivityType.playing.value,
            'name': str(self.name),
            'timestamps': timestamps
        }
        # fmt: on

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, ListeningActivity) and other.name == self.name

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self.name)
