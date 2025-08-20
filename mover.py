import pyautogui
import time
import random


def random_sleep(min_sec, max_sec):
    """Sleep for a random duration between min_sec and max_sec."""
    sleep_time = random.uniform(min_sec, max_sec)
    time.sleep(sleep_time)
    return sleep_time


def move_mouse():
    """Move mouse to a random location on screen."""
    screen_width, screen_height = pyautogui.size()
    x = random.randint(0, screen_width - 1)
    y = random.randint(0, screen_height - 1)
    pyautogui.moveTo(x, y, duration=random.uniform(0.2, 0.6))


def click_bottom_middle_third():
    """Click somewhere in the bottom middle third of the screen."""
    screen_width, screen_height = pyautogui.size()

    # X range = middle third of the screen width
    x_min = screen_width // 2-1
    x_max = screen_width // 2+1

    # Y range = bottom third of the screen height
    y_min = (screen_height * 2) // 3
    y_max = screen_height - 1

    # Pick a random safe spot within that zone
    x = random.randint(x_min, x_max)
    y = random.randint(y_min, y_max)

    pyautogui.click(x, y)


def main():
    last_click = time.time()
    next_click_delay = random.uniform(230, 250)  # 3m50s–4m10s

    while True:
        # Move mouse every 50–70 seconds
        random_sleep(50, 70)
        move_mouse()

        # Check if it's time for a click
        if time.time() - last_click >= next_click_delay:
            click_bottom_middle_third()
            last_click = time.time()
            next_click_delay = random.uniform(230, 250)  # re-schedule next click


if __name__ == "__main__":
    main()
