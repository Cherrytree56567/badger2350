import badger2040
import pngdec
import jpegdec

# Global Constants
WIDTH = badger2040.WIDTH
HEIGHT = badger2040.HEIGHT

IMAGE_WIDTH = 104

COMPANY_HEIGHT = 30
DETAILS_HEIGHT = 20
NAME_HEIGHT = HEIGHT - COMPANY_HEIGHT - (DETAILS_HEIGHT * 2) - 2
TEXT_WIDTH = WIDTH - IMAGE_WIDTH - 1

COMPANY_TEXT_SIZE = 0.6
DETAILS_TEXT_SIZE = 0.5

LEFT_PADDING = 5
NAME_PADDING = 20
DETAIL_SPACING = 10

BADGE_PATH = "/badges/badge.txt"

DEFAULT_TEXT = """mustelid inc
H. Badger
RP2040
2MB Flash
E ink
296x128px
/badges/badge.jpg
"""

# ------------------------------
#      Utility functions
# ------------------------------

# Reduce the size of a string until it fits within a given width
def truncatestring(text, text_size, width):
    while True:
        length = display.measure_text(text, text_size)
        if length > 0 and length > width:
            text = text[:-1]
        else:
            return text

# ------------------------------
#      Drawing functions
# ------------------------------

# Draw the badge, including user text
def draw_badge():
    display.set_pen(0)
    display.clear()


    # Draw a border around the image
    display.set_pen(0)
    display.line(WIDTH - IMAGE_WIDTH, 0, WIDTH - 1, 0)
    display.line(WIDTH - IMAGE_WIDTH, 0, WIDTH - IMAGE_WIDTH, HEIGHT - 1)
    display.line(WIDTH - IMAGE_WIDTH, HEIGHT - 1, WIDTH - 1, HEIGHT - 1)
    display.line(WIDTH - 1, 0, WIDTH - 1, HEIGHT - 1)

    # Draw the company name (top)
    display.set_pen(15)
    display.set_font("serif")
    display.text("CherryTree56567", LEFT_PADDING, (COMPANY_HEIGHT // 2) + 1, WIDTH, COMPANY_TEXT_SIZE)

    # Draw the first name and last name
    display.set_pen(15)
    display.set_font("serif")
    display.text("Ronit", LEFT_PADDING, NAME_PADDING, TEXT_WIDTH, 0.8)
    display.text("D'Silva", LEFT_PADDING, NAME_PADDING + 20, TEXT_WIDTH, 0.6)

    # Draw the job title
    display.set_pen(15)
    display.set_font("serif")
    job_title = truncatestring(jobtitle, DETAILS_TEXT_SIZE, TEXT_WIDTH)
    display.text("Developer", LEFT_PADDING, NAME_PADDING + 40, TEXT_WIDTH, DETAILS_TEXT_SIZE)

    # Draw the GitHub handle
    display.set_pen(15)
    githubhandle = "@" + githubhandle if not githubhandle.startswith('@') else githubhandle
    display.text("@CherryTree56567", LEFT_PADDING, NAME_PADDING + 60, TEXT_WIDTH, DETAILS_TEXT_SIZE)

    # Update the display
    display.update()
