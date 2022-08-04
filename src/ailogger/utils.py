import datetime


class UTCZone(datetime.tzinfo):
    """
    Free-standing implementation of the UTC timezone.

    This is implemented to provide some compatibility with older
    Python versions.

    """

    def tzname(self, dt):  # pragma: no cover -- required
        return "UTC"

    def utcoffset(self, dt):
        return datetime.timedelta(0)

    def dst(self, dt):
        return None

    def fromutc(self, dt):
        return dt


utc = UTCZone()
"""
UTC timezone instance.

Use this with :meth:`datetime.datetime.now` to produce a timezone
aware UTC timestamp.

"""


def utcnow():
    """
    Get a timezone aware UTC now.

    :returns: a timezone aware version of :func:`datetime.datetime.utcnow`
    :rtype: datetime.datetime

    """
    return datetime.datetime.now(tz=utc)
