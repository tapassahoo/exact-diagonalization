# modules/pkg_utils.py
import sys
import inspect
from termcolor import colored
from pkg_utils.config import *

def whoami():
	"""
	Prints the filename, function name, and line number from where `whoami()` is called, then exits.
	"""
	frame = inspect.currentframe().f_back

	# Separator
	print(colored("\n" + "=" * 80, SEPARATOR_COLOR))

	# Header
	print(colored("ATTENTION", DEBUG_COLOR, attrs=['bold', 'underline']))
	print(colored("\nCalled from:", HEADER_COLOR, attrs=['bold', 'underline']))

	# Information
	print(
		colored("File:".ljust(LABEL_WIDTH), LABEL_COLOR) +
		colored(f"{frame.f_code.co_filename}".ljust(VALUE_WIDTH), VALUE_COLOR)
	)
	print(
		colored("Function:".ljust(LABEL_WIDTH), LABEL_COLOR) +
		colored(f"{frame.f_code.co_name}".ljust(VALUE_WIDTH), VALUE_COLOR)
	)
	print(
		colored("Line:".ljust(LABEL_WIDTH), LABEL_COLOR) +
		colored(f"{frame.f_lineno}", VALUE_COLOR)
	)

	# Closing line
	print(colored("=" * 80, SEPARATOR_COLOR) + '\n')

	sys.exit(0)
