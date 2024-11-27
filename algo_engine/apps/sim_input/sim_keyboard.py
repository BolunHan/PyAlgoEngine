__package__ = 'algo_engine.apps.sim_input'

import ctypes
import enum
import time

from . import LOGGER, check_windows_version

check_windows_version((6, 1))
LOGGER.getChild('Keyboard')


# Constants for key event types
class KeyEvent(enum.IntEnum):
    F_KEYDOWN = 0x0000
    F_KEYUP = 0x0002


# Windows API functions
user32 = ctypes.windll.user32


# Map character to virtual key code and shift state
def get_keycode(char: str):
    """
    Get the virtual key code (VK_CODE) and shift state for a given character.

    Returns:
        tuple: (VK_CODE, needs_shift)
    """
    vk_combo = user32.VkKeyScanW(ord(char))
    vk_code = vk_combo & 0xFF  # Lower byte is VK_CODE
    shift_state = (vk_combo >> 8) & 0xFF  # Higher byte indicates shift state
    needs_shift = shift_state & 0x01  # Check if Shift key is required
    return vk_code, needs_shift


# Simulate key press
def press_key(vk_code):
    """Press a key using its VK_CODE."""
    ctypes.windll.user32.keybd_event(vk_code, 0, KeyEvent.F_KEYDOWN, 0)


# Simulate key release
def release_key(vk_code):
    """Release a key using its VK_CODE."""
    ctypes.windll.user32.keybd_event(vk_code, 0, KeyEvent.F_KEYUP, 0)


# Advanced function to simulate a key press for a given duration
def simulate_keypress(char, ts=0.1):
    """
    Simulate pressing a key for a given duration.

    Args:
        char (str): The character to simulate key press for.
        ts (float): The duration (in seconds) to hold the key. Default is 0.1s.
    """
    vk_code, needs_shift = get_keycode(char)  # Get VK_CODE and shift state
    if vk_code == 0xFF:
        raise ValueError(f"Invalid character '{char}'. Cannot map to a virtual key code.")

    # If the character needs Shift, press Shift first
    if needs_shift:
        press_key(0x10)  # VK_CODE for Shift

    press_key(vk_code)  # Press the key
    time.sleep(ts)  # Hold the key for the given duration
    release_key(vk_code)  # Release the key

    # If Shift was pressed, release it
    if needs_shift:
        release_key(0x10)  # VK_CODE for Shift


def main():
    LOGGER.info('Staring sim input sequence in 5 seconds...')
    time.sleep(5)
    key_strikes = list('hellow world!')
    for char in key_strikes:
        LOGGER.info(f'Pressing {char}...')
        simulate_keypress(char=char, ts=.5)
        time.sleep(.5)


# Example usage
if __name__ == "__main__":
    main()
