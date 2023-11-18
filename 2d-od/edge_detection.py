#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import math

import cv2 as cv
import numpy as np

# built-in module
import sys


def resize(img, fx, fy):
    return cv.resize(img, None, fx=fx, fy=fy)


def make_mask(img):
    h, w= img.shape
    rectangle = np.array([[0, h/3], [w, h/3], [w, h], [0, h]])

    mask = cv.fillPoly(np.zeros_like(img), np.int32([rectangle]), 255)
    mask = cv.bitwise_and(img, mask)
    return mask


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


def cured_line_detection(edges, min_line_length, max_line_gap):
    lines = cv.HoughLinesP(edges, cv.HOUGH_PROBABILISTIC, np.pi/180, 30, min_line_length, max_line_gap)
    return lines


def draw_curved_lines(img, lines, colour):
    for x in range(0, len(lines)):
        for x1, y1, x2, y2 in lines[x]:
            pts = np.array([[x1, y1], [x2, y2]], np.int32)
            cv.polylines(img, [pts], True, colour)


def straight_line_detection(edges):
    lines = cv.HoughLines(edges, cv.HOUGH_PROBABILISTIC, np.pi/180, 100)
    return lines


def draw_straight_line(img, lines, colour):
    for i in range(0, len(lines)):
        arr = np.array(lines[i][0], dtype=np.float64)
        r, theta = arr
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)


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
        prepped_img = make_mask(img_prep(img))

        # EDGE DETECTION 1 - CANNY EDGE DETECTION
        # Threshold values were selected based on previous testing
        edges = edge_detection(prepped_img, 2000, 4000)

        # EDGE DETECTION 2 - WHITE MASK
        white_img = white_mask(prepped_img)

        # straight_lines = straight_line_detection(edges)
        #
        # draw_straight_line(img, straight_lines, (255, 0, 0))

        # curved_lines = cured_line_detection(edges, 50, 5)
        #
        # black_img = np.zeros((700, 1000, 3), dtype=np.uint8)
        #
        # draw_curved_lines(black_img, curved_lines, (255, 0, 0))

        cnts = cv.findContours(white_mask(prepped_img), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            cv.drawContours(img, [c], -1, (255, 0, 0), thickness=5)

        # Colours for the overlay
        colours = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

        # Overlays edges and white mask to current frame
        vis = overlay_images(img, [edges], colours)

        # Displays frame with overlay
        cv.imshow('edge', vis)

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
