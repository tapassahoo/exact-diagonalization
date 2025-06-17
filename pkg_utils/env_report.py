# modules/env_report.py

import importlib
import logging
import os
import platform
import socket
import sys
import getpass
import pkg_resources
import subprocess
from datetime import datetime

# --- Set up logging ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = os.path.join(LOG_DIR, "env_report.log")
logging.basicConfig(level=logging.INFO,
					format="%(asctime)s - %(levelname)s - %(message)s",
					handlers=[
						logging.FileHandler(log_filename, mode='w'),
						logging.StreamHandler(sys.stdout)
					])

# --- List of required packages ---
REQUIRED_PACKAGES = [
	'numpy', 'scipy', 'matplotlib', 'pandas', 'qutip', 'sympy',
	'seaborn', 'sklearn', 'tensorflow', 'torch', 'numba'
]

def whom():
	"""
	Prints and logs basic system information.
	"""
	logging.info("="*60)
	logging.info("Execution Environment Info:")
	logging.info(f"User           : {getpass.getuser()}")
	logging.info(f"Hostname       : {socket.gethostname()}")
	try:
		ip_address = socket.gethostbyname(socket.gethostname())
		logging.info(f"IP Address     : {ip_address}")
	except socket.gaierror:
		logging.info("IP Address      : Not available")
	logging.info(f"System         : {platform.system()} {platform.release()}")
	logging.info(f"Architecture   : {platform.machine()}")
	logging.info(f"Python Version : {platform.python_version()}")
	logging.info(f"Working Dir    : {os.getcwd()}")
	logging.info(f"Timestamp      : {datetime.now()}")
	logging.info("="*60)

def check_installed_packages():
	"""
	Logs which scientific packages are installed and their versions.
	"""
	logging.info("\nChecking installed scientific packages...")
	for pkg in REQUIRED_PACKAGES:
		try:
			version = pkg_resources.get_distribution(pkg).version
			logging.info(f"[✔] {pkg:<12} - version {version}")
		except pkg_resources.DistributionNotFound:
			logging.warning(f"[✘] {pkg:<12} - NOT INSTALLED")

def safe_import(package_name):
	"""
	Attempts to import a package safely.
	Returns the module object if successful, or None if failed.
	"""
	try:
		return importlib.import_module(package_name)
	except ImportError:
		logging.warning(f"Could not import package: {package_name}")
		return None

def list_available(required_list=None):
	"""
	Lists available packages from a given list or default list.
	"""
	logging.info("\nAvailable packages:")
	packages = required_list if required_list else REQUIRED_PACKAGES
	for pkg in packages:
		module = safe_import(pkg)
		if module:
			logging.info(f"[✔] {pkg}")
		else:
			logging.warning(f"[✘] {pkg}")

def install_missing_packages():
	"""
	Attempts to install missing packages via pip.
	"""
	logging.info("\nAttempting to install missing packages...")
	for pkg in REQUIRED_PACKAGES:
		try:
			pkg_resources.get_distribution(pkg)
		except pkg_resources.DistributionNotFound:
			logging.warning(f"Installing {pkg}...")
			try:
				subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
				logging.info(f"[✔] Successfully installed {pkg}")
			except subprocess.CalledProcessError:
				logging.error(f"[✘] Failed to install {pkg}")
