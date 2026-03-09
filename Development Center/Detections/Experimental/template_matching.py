import os
import cv2 as cv
import numpy as np


example_image_path = (
    os.path.dirname(os.path.realpath(__file__)) + "/../../../Images/Examples/example.png"
)
door_image_path = (
    os.path.dirname(os.path.realpath(__file__)) + "/../../../Images/Models/Doors/door.png"
)
window_image_path = (
    os.path.dirname(os.path.realpath(__file__)) + "/../../../Images/Models/Windows/window.png"
)



def match(image, template, name, threshold=0.99):
    img_rgb = image
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

    test(img_gray, template, cv.TM_CCOEFF_NORMED, threshold, "TM_CCOEFF_NORMED " + name)
    test(img_gray, template, cv.TM_CCORR_NORMED, threshold, "TM_CCORR_NORMED" + name)
    test(img_gray, template, cv.TM_SQDIFF_NORMED, threshold, "TM_SQDIFF_NORMED" + name)
    cv.imshow(name, template)


def test(img_gray, template, alg, threshold, name):
    res = cv.matchTemplate(img_gray, template, alg)
    loc = np.where(res >= threshold)
    w, h = template.shape[::-1]
    show(loc, name, w, h)


def show(loc, name, w, h, max=100):
    i = 0
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        if i > max:
            print("Max " + str(max) + " reached!")
            break
        i += 1

    cv.imshow(name, img_rgb)
    cv.waitKey(0)


img_rgb = cv.imread(example_image_path)

window_template = cv.imread(window_image_path, 0)
door_template = cv.imread(door_image_path, 0)

print("window")
match(img_rgb, window_template, "window")
print("door")
match(img_rgb, door_template, "door")
print("text")
