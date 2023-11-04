#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import numpy as np

# built-in module
import sys


def resize(img, fx, fy):
    return cv.resize(img, None, fx=fx, fy=fy)


def img_prep(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return cv.GaussianBlur(gray, (5, 5), 0)


def blur(img):
    return cv.GaussianBlur(img, (5, 5), 0)


def edge_detection(img, threshold1, threshold2):
    return cv.Canny(img, threshold1, threshold2, apertureSize=5)


def white_mask(img):
    white_img = cv.inRange(img, 200, 240)
    return cv.bitwise_and(img, white_img)


def overlay_images(img, overlays, colours):
    """
    Overlays images in on an image with a specific colour

    Precondition:
    - len(overlays) <= len(colours)
    """
    vis = img.copy()

    # Dimmer video
    vis = np.uint8(vis/2.)

    for i in range(0, len(overlays)):
        vis[overlays[i] != 0] = colours[i]

    return vis


def main():
    cv.namedWindow('edge')
    # cv.createTrackbar('thrs1', 'edge', 2000, 5000, nothing)
    # cv.createTrackbar('thrs2', 'edge', 4000, 5000, nothing)

    cap = cv.VideoCapture("../20230605_163120.mp4")

    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    end = False

    curr_frame = 0
    while curr_frame < int(frame_count) and not end:
        # To speed process, the software only runs on every 8th frame
        # Possible implementation of only checking every nth millisecond/second
        for i in range(7):
            cap.read()

        # Gets current frame info
        _flag, img = cap.read()

        # Resizes image (as original video is too lage)
        img = resize(img, 0.25, 0.25)

        # Applies greyscale and gaussian blur
        prepped_img = img_prep(img)

        # EDGE DETECTION 1 - CANNY EDGE DETECTION
        # Threshold values were selected based on previous testing
        edges = edge_detection(prepped_img, 2000, 4000)

        # EDGE DETECTION 2 - WHITE MASK
        white_img = white_mask(prepped_img)

        # Overlays edge detection 1 and 2 to current frame
        # vis = img.copy()
        # vis = np.uint8(vis/2.)
        # vis[edges != 0] = (0, 255, 0)
        # vis[white_img != 0] = (0, 0, 255)

        # Colours for the overlay
        colours = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

        # Overlays edges and white mask to current frame
        vis = overlay_images(img, [edges, white_img], colours)

        # Displays frame with overlay
        cv.imshow('edge', vis)

        # vis3 = mask_white_image.copy()
        # vis3 = np.uint8(vis3 / 2.)
        # vis3[edge != 0] = (0, 0, 255)

        # cv.imshow('edge', vis3)

        key = cv.waitKey(1)
        if key == 27:
            break
        curr_frame += 1

    print('Reached end of video')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
