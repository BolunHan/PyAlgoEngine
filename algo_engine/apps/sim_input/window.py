__package__ = 'algo_engine.apps.sim_input'

import ctypes
import dataclasses
import enum
import os
from typing import Literal

from . import LOGGER

LOGGER.getChild('Window')

# Define necessary Windows API functions and constants
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32
psapi = ctypes.windll.psapi

# Define constants for access rights
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010

# Define necessary Windows API function prototypes
WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.POINTER(ctypes.c_int))


@dataclasses.dataclass(frozen=True)
class WindowInfo:
    window_name: str
    pid: int
    hwnd: int
    executable_name: str
    executable_path: str


class WindowState(enum.IntEnum):
    SW_SHOWNORMAL = 1  # Show the window normally
    SW_SHOWMINIMIZED = 2  # Minimize the window
    SW_SHOWMAXIMIZED = 3  # Maximize the window
    SW_SHOWNOACTIVATE = 4  # Show the window without activating it
    SW_SHOW = 5  # Show the window and bring it to the foreground


# Function to get PID from window handle
def get_pid(hwnd):
    pid = ctypes.c_ulong()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


# Function to retrieve the executable path and name of a process
def get_executable_info(pid):
    # Open the process to get information
    h_process = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
    if not h_process:
        return None, None

    # Buffer to hold the path
    path_buffer = ctypes.create_unicode_buffer(1024)

    # Get the full executable path using GetModuleFileNameEx
    if psapi.GetModuleFileNameExW(h_process, 0, path_buffer, ctypes.byref(ctypes.c_ulong(1024))):
        executable_path = path_buffer.value
        executable_name = os.path.basename(executable_path)
    else:
        executable_path = None
        executable_name = None

    # Close the process handle
    kernel32.CloseHandle(h_process)

    return executable_name, executable_path


# Function to check if a window is visible (including minimized)
def is_window_visible(hwnd):
    return user32.IsWindowVisible(hwnd)


# Function to enumerate windows
def get_windows() -> list[WindowInfo]:
    windows = []

    def enum_windows_proc(hwnd, _):
        # Only include visible windows
        if not is_window_visible(hwnd):
            return True

        # Get window title (for name)
        length = user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            window_name = buffer.value
        else:
            window_name = "Untitled"

        # Get the PID for the window
        pid = get_pid(hwnd)

        # Get executable name and path
        executable_name, executable_path = get_executable_info(pid)

        # Append to the windows list as a tuple (window_name, pid, hwnd, executable_name, executable_path)
        windows.append(WindowInfo(window_name=window_name, pid=pid, hwnd=hwnd, executable_name=executable_name, executable_path=executable_path))
        return True

    # Enumerate all windows (including child windows)
    user32.EnumWindows(WNDENUMPROC(enum_windows_proc), 0)

    return windows


def find_window(name: str = None, executable: str = None) -> list[WindowInfo]:
    windows = get_windows()
    matched = []

    for window in windows:
        if name is not None and name.lower() not in window.name.lower():
            continue

        if executable is not None and executable.lower() not in window.executable_name.lower():
            continue

        matched.append(window)

    return matched


# Function to set the window action (top, maximize, minimize)
def set_window(window_info: WindowInfo, action: Literal['top', 'maximize', 'minimize', 'max', 'min']):
    hwnd = window_info.hwnd

    match action:
        case "top":
            # Bring the window to the front (top)
            user32.ShowWindow(hwnd, WindowState.SW_SHOWNORMAL)
            user32.SetForegroundWindow(hwnd)
        case "maximize" | 'max':
            # Maximize the window
            user32.ShowWindow(hwnd, WindowState.SW_SHOWMAXIMIZED)
        case "minimize" | 'min':
            # Minimize the window
            user32.ShowWindow(hwnd, WindowState.SW_SHOWMINIMIZED)
        case _:
            raise ValueError(f"Unknown action: {action}")


def main():
    # Example usage
    windows = get_windows()
    for _ in windows:
        LOGGER.debug(_)

    firefox = find_window(executable="firefox")[0]
    LOGGER.info(firefox)

    set_window(window_info=firefox, action='top')
    set_window(window_info=firefox, action='maximize')


if __name__ == "__main__":
    main()
