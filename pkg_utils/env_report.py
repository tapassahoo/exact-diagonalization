import importlib
import importlib.util
import logging
import subprocess
import sys
import argparse
import os
import socket
import platform
import getpass
from datetime import datetime

REQUIRED_PACKAGES = [
	"argparse", "os", "inspect", "sys", "getpass", "socket", "platform",
	"math", "cmath", "numpy", "scipy", "datetime", "termcolor", "typing",
	"pandas", "matplotlib", "seaborn", "netCDF4", "contextlib", "io"
]

def setup_logging():
	logging.basicConfig(
		level=logging.INFO,
		format="%(levelname)-8s: %(message)s",
		handlers=[logging.StreamHandler(sys.stdout)]
	)

def is_module_available(pkg_name: str) -> bool:
	return importlib.util.find_spec(pkg_name) is not None

def check_installed_packages(package_list=None):
	package_list = package_list or REQUIRED_PACKAGES
	logging.info("\n📦 Checking package availability:")
	for pkg in package_list:
		if is_module_available(pkg):
			logging.info(f"  [✔] {pkg:<12} - AVAILABLE")
		else:
			logging.warning(f"  [✘] {pkg:<12} - NOT FOUND")

def safe_import(pkg_name: str):
	try:
		return importlib.import_module(pkg_name)
	except ImportError:
		logging.warning(f"[✘] Import failed: {pkg_name}")
		return None

def list_available(package_list=None):
	package_list = package_list or REQUIRED_PACKAGES
	logging.info("\n📃 Listing available modules:")
	for pkg in package_list:
		if is_module_available(pkg):
			logging.info(f"  [✔] {pkg}")
		else:
			logging.warning(f"  [✘] {pkg}")

def install_missing_packages(package_list=None):
	package_list = package_list or REQUIRED_PACKAGES
	logging.info("\n🔧 Attempting to install missing packages...")
	for pkg in package_list:
		if not is_module_available(pkg):
			logging.warning(f"  [✘] {pkg} missing. Trying to install...")
			try:
				subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
				logging.info(f"  [✔] Installed {pkg}")
			except subprocess.CalledProcessError:
				logging.error(f"  [✘] Failed to install {pkg}")
		else:
			logging.debug(f"  [✔] {pkg} already available")

def whom():
	"""
	Prints and logs basic system information.
	"""
	print("\n")
	logging.info("="*60)
	logging.info("Execution Environment Info:")
	logging.info(f"User           : {getpass.getuser()}")
	logging.info(f"Hostname       : {socket.gethostname()}")
	try:
		ip_address = socket.gethostbyname(socket.gethostname())
		logging.info(f"IP Address     : {ip_address}")
	except socket.gaierror:
		logging.info("IP Address     : Not available")
	logging.info(f"System         : {platform.system()} {platform.release()}")
	logging.info(f"Architecture   : {platform.machine()}")
	logging.info(f"Python Version : {platform.python_version()}")
	logging.info(f"Working Dir    : {os.getcwd()}")
	logging.info(f"Timestamp      : {datetime.now()}")
	logging.info("="*60)

def main():
	parser = argparse.ArgumentParser(description="Check, list, and install Python packages.")
	parser.add_argument("--check", action="store_true", help="Check which packages are available")
	parser.add_argument("--install", action="store_true", help="Install missing packages")
	parser.add_argument("--list", action="store_true", help="List all importable packages")
	parser.add_argument("--info", action="store_true", help="Display system environment info")

	args = parser.parse_args()
	setup_logging()

	if args.info:
		whom()
	if args.check:
		check_installed_packages()
	if args.install:
		install_missing_packages()
	if args.list:
		list_available()

	if not any(vars(args).values()):
		parser.print_help()

if __name__ == "__main__":
	main()

