from ailogger.filters import AttributeSetter
from ailogger.formatters import AIJsonFormatter

version_info = (2, 0, 1)
version = ".".join(str(c) for c in version_info)

__all__ = [
    "AttributeSetter",
    "AIJsonFormatter",
    "version",
    "version_info",
]
