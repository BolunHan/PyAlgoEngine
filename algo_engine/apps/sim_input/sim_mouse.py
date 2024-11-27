__package__ = 'algo_engine.apps.sim_input'

import ctypes
import enum
import time
from ctypes import wintypes
from typing import Literal

from . import LOGGER, check_windows_version

check_windows_version((6, 1))
LOGGER.getChild('Mouse')


# Constants for mouse events
class MouseEvent(enum.IntEnum):
    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_ABSOLUTE = 0x8000


# Structure to store point coordinates
class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


# Windows API functions
user32 = ctypes.windll.user32


# Get the current mouse location
def get_location() -> POINT:
    """
    Get the current mouse cursor position.

    Returns:
        tuple: (x, y) coordinates of the mouse cursor.
    """
    point = POINT()
    user32.GetCursorPos(ctypes.byref(point))
    return point


def move_mouse(x: int, y: int):
    """
    Move the mouse cursor to a specific screen location.

    Args:
        x (int): The target x-coordinate.
        y (int): The target y-coordinate.
    """
    user32.SetCursorPos(x, y)


# Simulate mouse press
def press_mouse(button: Literal['l', 'r', 'left', 'right'] = 'l'):
    """
    Simulate a mouse button press.

    Args:
        button (str): The button to press ("left" or "right").
    """
    match button:
        case 'l' | "left":
            user32.mouse_event(MouseEvent.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        case 'r' | "right":
            user32.mouse_event(MouseEvent.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        case _:
            raise ValueError("Invalid button. Use 'left' or 'right'.")


# Simulate mouse release
def release_mouse(button: Literal['l', 'r', 'left', 'right'] = 'l'):
    """
    Simulate a mouse button release.

    Args:
        button (str): The button to release ("left" or "right").
    """

    match button:
        case 'l' | "left":
            user32.mouse_event(MouseEvent.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        case 'r' | "right":
            user32.mouse_event(MouseEvent.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
        case _:
            raise ValueError("Invalid button. Use 'left' or 'right'.")


# Simulate a mouse click
def click_mouse(button: Literal['l', 'r', 'left', 'right'] = 'l'):
    """
    Simulate a mouse click (press + release).

    Args:
        button (str): The button to click ("left" or "right").
    """
    press_mouse(button)
    release_mouse(button)


# Simulate a mouse double click
def double_click_mouse(button: Literal['l', 'r', 'left', 'right'] = 'l', interval: float = 0.1):
    """
    Simulate a mouse double click.

    Args:
        button (str): The button to double-click ("left" or "right").
        interval (float): Time between clicks (in seconds). Default is 0.1s.
    """
    click_mouse(button)
    time.sleep(interval)
    click_mouse(button)


# Example usage
if __name__ == "__main__":
    LOGGER.info('Staring sim mouse input sequence in 5 seconds...')
    time.sleep(5)

    p = get_location()
    LOGGER.info(f"Current mouse location: {p.x, p.y}")

    LOGGER.info('Move to diagonal location...')
    move_mouse(x=p.y, y=p.x)

    LOGGER.info(f"Simulating left click...")
    click_mouse(button="l")

    LOGGER.info(f"Simulating right click...")
    click_mouse(button="r")

    LOGGER.info("Simulating double click...")
    double_click_mouse()
