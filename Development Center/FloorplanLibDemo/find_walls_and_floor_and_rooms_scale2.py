import pytest
import cv2
import numpy as np
import sys

import os

floorplan_lib_path = os.path.dirname(os.path.realpath(__file__)) + "/../../"
example_image_path = (
    os.path.dirname(os.path.realpath(__file__)) + "/../../Images/Examples/example.png"
)

try:
    sys.path.insert(0, floorplan_lib_path)
    from FloorplanToBlenderLib import *
except ImportError:
    from FloorplanToBlenderLib import *

from subprocess import check_output
import os


def test(path):
    img = cv2.imread(path)
    image = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width, channels = img.shape
    blank_image = np.zeros(
        (height, width, 3), np.uint8
    )

    wall_img = detect.wall_filter(gray)
    wall_temp = wall_img
    boxes, img = detect.precise_boxes(wall_img, blank_image)

    gray = ~wall_temp

    doors, colored_doors = detect.find_details(
        gray.copy(),
        noise_removal_threshold=50,
        corners_threshold=0.1,
        room_closing_max_length=150,
        gap_in_wall_max_threshold=2000,
        gap_in_wall_min_threshold=100,
    )
    gray_details = cv2.cvtColor(colored_doors, cv2.COLOR_BGR2GRAY)
    boxes, blank_image = detect.precise_boxes(
        gray_details, blank_image, color=(0, 200, 100)
    )

    cv2.imshow("detection", blank_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


test(example_image_path)
