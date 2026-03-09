import cv2
import numpy as np
import matplotlib.pyplot as plt




def image(image, title="FTBL", wait=0):

    cv2.imshow(title, image)
    cv2.waitKey(wait)


def points(image, points):

    for point in points:
        image = cv2.circle(image, point, radius=4, color=(0, 0, 0), thickness=5)
    return image


def contours(image, contours):

    return cv2.drawContours(image, contours, -1, (0, 255, 0), 3)


def lines(image, lines):

    for line in lines:
        image = cv2.polylines(image, line, True, (0, 0, 255), 1, cv2.LINE_AA)
    return image


def verts(image, boxes):

    for box in boxes:
        for wall in box:

            cv2.line(
                image,
                (int(wall[0][0]), int(wall[1][1])),
                (int(wall[2][0]), int(wall[2][1])),
                (255, 0, 0),
                5,
            )


def boxes(image, boxes, text=""):

    for box in boxes:
        (x, y, w, h) = cv2.boundingRect(box)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)
        cv2.putText(image, str(text), (x, y), 7, 10, (255, 0, 0))
    return image


def doors(img, doors):

    for door in doors:
        img = points(img, door[0])
        img = boxes(img, door[1])
    return img


def colormap(img, mapping=cv2.COLORMAP_HSV):

    return cv2.applyColorMap(img, mapping)


def histogram(img, title="Histogram", wait=0):

    hist = np.histogram(img, bins=np.arange(0, 256))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.imshow(img, cmap=plt.cm.gray, interpolation="nearest")
    ax1.axis("off")
    ax2.plot(hist[1][:-1], hist[0], lw=2)
    ax2.set_title(title)
    if wait == 0:
        plt.show()
    else:
        plt.pause(wait)
