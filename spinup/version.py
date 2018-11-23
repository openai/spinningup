version_info = (0, 1, 2)
# format:
# ('spinup_major', 'spinup_minor', 'spinup_patch')

def get_version():
    "Returns the version as a human-format string."
    return '%d.%d.%d' % version_info

__version__ = get_version()