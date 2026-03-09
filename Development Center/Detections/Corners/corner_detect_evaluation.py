import cv2
import numpy as np
import os

example_image_path = (
    os.path.dirname(os.path.realpath(__file__)) + "/../../../Images/Examples/example.png"
)


img = cv2.imread(example_image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

img_canny = cv2.Canny(img, 100, 200)

img_sobelx = cv2.Sobel(img_gaussian, cv2.CV_8U, 1, 0, ksize=5)
img_sobely = cv2.Sobel(img_gaussian, cv2.CV_8U, 0, 1, ksize=5)
img_sobel = img_sobelx + img_sobely

kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
laplacian = cv2.Laplacian(img, cv2.CV_64F)

cv2.imshow("Original Image", img)
cv2.imshow("Canny", img_canny)
cv2.imshow("Sobel X", img_sobelx)
cv2.imshow("Sobel Y", img_sobely)
cv2.imshow("Sobel", img_sobel)
cv2.imshow("Prewitt X", img_prewittx)
cv2.imshow("Prewitt Y", img_prewitty)
cv2.imshow("Prewitt", img_prewittx + img_prewitty)
cv2.imshow("laplacian", laplacian)

cv2.waitKey(0)
cv2.destroyAllWindows()
