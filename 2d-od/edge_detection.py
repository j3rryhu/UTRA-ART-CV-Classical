#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

from typing import Any

import cv2 as cv
import numpy as np

from timeit import default_timer as timer


def resize(img, fx, fy) -> np.ndarray[Any, np.dtype[np.generic]]:
    """
    Resizes the image by a factor of fx horizontally and fy vertically and returns an image

    Arguments:
    img - Image being resized
    fx - X scale
    fy = Y scale
    """
    return cv.resize(img, None, fx=fx, fy=fy)


def img_prep(img: np.ndarray[Any, np.dtype[np.generic]]) -> np.ndarray[Any, np.dtype[np.generic]]:
    """
    Converts image to greyscale and blurred version of that image and returns it

    - img - Image being converted
    """
    gray = grayscale(img)
    return blur(gray)


def grayscale(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def blur(img):
    return cv.GaussianBlur(img, (5, 5), 0)


def make_ver_mask(img: np.ndarray[Any, np.dtype[np.generic]]):
    """


    :param img:
    :return:
    """
    shape = img.shape
    h = shape[0]
    w = shape[1]
    rectangle = np.array([[0, h/3], [w, h/3], [w, h], [0, h]])

    mask = cv.fillPoly(np.zeros_like(img), np.int32([rectangle]), 255)
    mask = cv.bitwise_and(img, mask)
    return mask


def edge_detection(img, threshold1, threshold2):
    return cv.Canny(img, threshold1, threshold2, apertureSize=5)


def white_mask(img):
    white_img = cv.inRange(img, 200, 240)
    gray = grayscale(img)
    return cv.bitwise_and(gray, white_img)


def combine_images_alpha(front, background):
    # Extracts alpha channel from transparent image as mask
    alpha = front[:, :, 3]
    alpha = cv.merge([alpha, alpha, alpha])

    # Extracts bgr channels from transparent image
    front_bgr = front[:, :, 0:3]

    # Blend the two images using the alpha channel as controlling mask
    return np.where(alpha == (0, 0, 0), background, front_bgr)


def canny_edge_detection(prepped_img):
    # Threshold values were selected based on previous testing
    edges = edge_detection(prepped_img, 1000, 3500)

    # Draws edges on transparent image
    edges_transparent = np.zeros((540, 960, 4), dtype=np.uint8)
    edges_transparent[edges != 0] = (255, 0, 0, 255)
    return edges_transparent


def white_detection(prepped_img):
    contours = cv.findContours(white_mask(prepped_img), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Draws contours on transparent image
    contours_transparent = np.zeros((540, 960, 4), dtype=np.uint8)
    cv.drawContours(contours_transparent, contours, -1, (255, 0, 0, 255),
                    10)  # change first three channels to any color you want.

    return contours_transparent


def main():
    cv.namedWindow('edge')

    cap = cv.VideoCapture("../20230605_163120.mp4")

    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    end = False

    curr_frame = 0
    while (curr_frame + 2) < int(frame_count) and not end:
        start_time = timer()

        # To speed process, the software only runs on every 8th frame
        # Possible implementation of only checking every nth millisecond/second
        # for i in range(2):
        #     cap.grab()

        cap.read()

        # Gets current frame info
        _flag, img = cap.read()

        # Resizes image (as original video is too lage)
        img = resize(img, 0.25, 0.25)

        # Applies greyscale and gaussian blur
        prepped_img = make_ver_mask(img_prep(img))

        # EDGE DETECTION 1 - CANNY EDGE
        edges_transparent = canny_edge_detection(prepped_img)

        # EDGE DETECTION 2 - WHITE MASK
        other_prep = make_ver_mask(blur(img))
        contours_transparent = white_detection(other_prep)

        # Combines EDGE DETECTION 1 & 2
        combined_detection = cv.bitwise_and(contours_transparent, edges_transparent)

        # Overlays combined edge detections with current video frame
        result = combine_images_alpha(combined_detection, img)

        # EXTRA RESULTS OF EDGE DETECTION 1 AND 2 ALONE
        # result_white = combine_images_alpha(contours_transparent, img)
        # result_canny = combine_images_alpha(edges_transparent, img)

        # Displays video frame with edge detection overlaid
        cv.imshow('edge', result)

        end_time = timer()
        print("Time to load frame: " + (str)(end_time - start_time))

        # Checks to see if ESC key is pressed
        key = cv.waitKey(1)
        if key == 27:
            break
        curr_frame += 1

    print('Reached end of video')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
