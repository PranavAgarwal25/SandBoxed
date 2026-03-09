import cv2
import numpy as np
import sys
import os

floorplan_lib_path = os.path.dirname(os.path.realpath(__file__)) + "/../../../"
example_image_path = (
    os.path.dirname(os.path.realpath(__file__)) + "/../../../Images/Examples/example.png"
)

sys.path.insert(0, floorplan_lib_path)
from FloorplanToBlenderLib import *
from subprocess import check_output
import os


def test():
    img = cv2.imread(example_image_path)
    image = img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width, channels = img.shape
    blank_image = np.zeros(
        (height, width, 3), np.uint8
    )

    wall_img = detect.wall_filter(gray)

    res, out = detect.and_remove_precise_boxes(wall_img, output_img=gray)

    verts = []
    faces = []

    height = 0

    img = cv2.medianBlur(gray, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cimg = image

    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30
    )

    cv2.imshow("detected circles", img)
    cv2.imshow("detected circ2s", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Test Done!")


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 3:
            shape = "triangle"

        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        elif len(approx) == 5:
            shape = "pentagon"

        else:
            shape = "circle"

        return shape


if __name__ == "__main__":
    test()


img = cv2.imread(example_image_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

wall_img = detect.wall_filter(gray)

res, out = detect.and_remove_precise_boxes(wall_img, output_img=gray)


gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 50, 250)
kernel = cv2.getStructuringElement(2, (6, 6))
closed = cv2.morphologyEx(edged, 3, kernel)
(cnts, _) = cv2.findContours(closed.copy(), 0, 1)
total = 0
for c in cnts:
    peri = cv2.arcLength(c, True)

    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    area = cv2.contourArea(c)

    if len(approx) >= 4 and area < 2000 and area > 450:
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
        total += 1

cv2.imshow("Gray", img)
cv2.imshow("Gwray", gray)

cv2.waitKey(0)
