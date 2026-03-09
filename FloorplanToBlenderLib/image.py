import cv2
import numpy as np
from PIL import Image
from . import calculate
from . import const




def pil_rescale_image(image, factor):
    width, height = image.size
    return image.resize((int(width * factor), int(height * factor)), resample=Image.BOX)


def cv2_rescale_image(image, factor):
    return cv2.resize(image, None, fx=factor, fy=factor)


def pil_to_cv2(image):
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def calculate_scale_factor(preferred: float, value: float):
    return preferred / value


def denoising(img):
    return cv2.fastNlMeansDenoisingColored(
        img,
        None,
        const.IMAGE_H,
        const.IMAGE_HCOLOR,
        const.IMAGE_TEMPLATE_SIZE,
        const.IMAGE_SEARCH_SIZE,
    )


def remove_noise(img, noise_removal_threshold):

    img[img < 128] = 0
    img[img > 128] = 255
    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(img)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > noise_removal_threshold:
            cv2.fillPoly(mask, [contour], 255)
    return mask


def mark_outside_black(img, mask):


    contours, _ = cv2.findContours(~img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    mask = np.zeros_like(mask)
    cv2.fillPoly(mask, [biggest_contour], 255)
    img[mask == 0] = 0
    return img, mask


def detect_wall_rescale(reference_size, image):

    image_wall_size = calculate.wall_width_average(image)
    if image_wall_size is None:
        return None
    return calculate_scale_factor(float(reference_size), image_wall_size)
