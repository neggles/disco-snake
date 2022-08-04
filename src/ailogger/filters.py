import logging
import uuid

from ailogger import utils


class AttributeSetter(logging.Filter):
    """
    Ensure that attributes exist on :class:`~logging.LogRecord` s.

    :keyword dict add_fields: maps fields to create on
        :class:`~logging.LogRecord` instances to their default
        values

    The values in the `add_fields` mapping can be strings that start
    with ``'ext://'`` to invoke custom behaviors.  The following values
    are recognized:

    **UUID**
        Generate a new UUIDv4 instance via :func:`uuid.uuid4()`

    **now**
        Generate a new timezone-aware UTC :class:`datetime.datetime`
        instance

    """

    def __init__(self, *args, **kwargs):
        self.add_fields = kwargs.pop("add_fields", {})
        logging.Filter.__init__(self, *args, **kwargs)
        self.ext_map = {
            "ext://UUID": uuid.uuid4,
            "ext://now": utils.utcnow,
        }

    def filter(self, record):
        for name, default in self.add_fields.items():
            if not hasattr(record, name):
                try:
                    setattr(record, name, self.ext_map[default]())
                except KeyError:
                    setattr(record, name, default)
        return 1
