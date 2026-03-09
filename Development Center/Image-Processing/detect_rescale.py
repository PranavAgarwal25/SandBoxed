import cv2
import numpy as np
import os
import sys

try:
    sys.path.insert(1, sys.path[0] + "/../..")
    from FloorplanToBlenderLib import *
except ImportError as e:
    print(e)
    raise ImportError


def main():
    path = os.path.dirname(os.path.realpath(__file__)) + "/../../Images/Examples/example.png"
    img = cv2.imread(path)
    preferred = calculate_wall_width_average(img)

    path = os.path.dirname(os.path.realpath(__file__)) + "/../../Images/Examples/example2.png"
    img = cv2.imread(path)
    too_small1 = calculate_wall_width_average(img)

    scalefactor1 = calculate_scale_factor(preferred, too_small1)

    path = os.path.dirname(os.path.realpath(__file__)) + "/../../Images/Examples/example3.png"
    img = cv2.imread(path)
    too_small2 = calculate_wall_width_average(img)

    scalefactor2 = calculate_scale_factor(preferred, too_small2)

    print("The preferred pixel size per wall is : ", preferred)
    print("Example image 2 should be scaled by : ", scalefactor1)
    print("Example image 3 should be scaled by : ", scalefactor2)


def calculate_scale_factor(preferred, value):
    return preferred / value


def calculate_wall_width_average(img):
    image = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width, channels = img.shape
    blank_image = np.zeros(
        (height, width, 3), np.uint8
    )

    wall_img = detect.wall_filter(gray)
    wall_temp = wall_img
    boxes, img = detect.detectPreciseBoxes(wall_img, blank_image)

    filtered_boxes = list()
    for box in boxes:
        if len(box) == 4:
            x, y, w, h = cv2.boundingRect(box)
            if w > h:
                shortest = h
            else:
                shortest = w
            filtered_boxes.append(shortest)

    return Average(filtered_boxes)


def Average(lst):
    return sum(lst) / len(lst)


if __name__ == "__main__":
    main()
