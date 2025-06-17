# pkg_utils/config.py

# Color constants
HEADER_COLOR = 'cyan'
LABEL_COLOR = 'green'
VALUE_COLOR = 'magenta'
DEBUG_COLOR = 'red'
SEPARATOR_COLOR = 'yellow'
INFO_COLOR = 'blue'

# Widths for formatting
LABEL_WIDTH = 35
VALUE_WIDTH = 45

# Make these available when using `from config import *`
__all__ = [
	"HEADER_COLOR", "LABEL_COLOR", "VALUE_COLOR", "DEBUG_COLOR",
	"SEPARATOR_COLOR", "INFO_COLOR", "LABEL_WIDTH", "VALUE_WIDTH"
]

