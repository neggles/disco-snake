try:
    from disco_snake._version import version as __version__
    from disco_snake._version import version_tuple
except ImportError:
    __version__ = "unknown (no version information available)"
    version_tuple = (0, 0, "unknown", "noinfo")
