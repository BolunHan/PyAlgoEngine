import platform

from .. import LOGGER


def check_windows_version(min_version=(6, 1)):
    """
    Ensure the script is running on a compatible Windows version.
    :param min_version: Minimum required version as a tuple (major, minor).
                        Default is (6, 1) for Windows 7 (major=6, minor=1).
    """
    # Get the Windows version
    if not platform.system() == "Windows":
        raise EnvironmentError("This script only works on Windows.")

    version_str = platform.version()
    version_tuple = tuple(map(int, platform.win32_ver()[1].split('.')))

    if version_tuple < min_version:
        raise EnvironmentError(f"Unsupported Windows version: {version_str}. Minimum required is {min_version[0]}.{min_version[1]}.")


LOGGER = LOGGER.getChild('SimInput')
