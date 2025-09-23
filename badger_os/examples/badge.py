import os
import badger2040
from badger2040 import HEIGHT, WIDTH
import badger_os
import jpegdec
import pngdec


TOTAL_IMAGES = 0


# Turn the act LED on as soon as possible
display = badger2040.Badger2040()
display.led(128)
display.set_update_speed(badger2040.UPDATE_NORMAL)

jpeg = jpegdec.JPEG(display.display)
png = pngdec.PNG(display.display)


# Load images
try:
    IMAGES = [f for f in os.listdir("/badges") if f.endswith(".jpg") or f.endswith(".png")]
    TOTAL_IMAGES = len(IMAGES)
except OSError:
    pass


state = {
    "current_image": 0,
    "show_info": True
}


def draw_badge(n):
    file = IMAGES[n]
    name, ext = file.split(".")

    try:
        png.open_file("/badges/{}".format(file))
        png.decode()
    except (OSError, RuntimeError):
        jpeg.open_file("/badges/{}".format(file))
        jpeg.decode()

    display.update()


badger_os.state_load("image", state)

changed = True


while True:
    # Sometimes a button press or hold will keep the system
    # powered *through* HALT, so latch the power back on.
    display.keepalive()

    if display.pressed(badger2040.BUTTON_UP):
        if state["current_image"] > 0:
            state["current_image"] -= 1
            changed = True

    if display.pressed(badger2040.BUTTON_DOWN):
        if state["current_image"] < TOTAL_IMAGES - 1:
            state["current_image"] += 1
            changed = True

    if display.pressed(badger2040.BUTTON_A):
        state["show_info"] = not state["show_info"]
        changed = True

    if changed:
        draw_badge(state["current_image"])
        badger_os.state_save("image", state)
        changed = False

    # Halt the Badger to save power, it will wake up if any of the front buttons are pressed
    display.halt()
