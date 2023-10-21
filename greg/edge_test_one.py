#!/usr/bin/env python

'''
This sample demonstrates Canny edge detection.

Usage:
  edge.py [<video source>]

  Trackbars control edge thresholds.

'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import numpy as np

# built-in module
import sys


def main():
    try:
        fn = sys.argv[1]
    except:
        fn = 0

    def nothing(*arg):
        pass

    cv.namedWindow('edge')
    # cv.createTrackbar('thrs1', 'edge', 2000, 5000, nothing)
    # cv.createTrackbar('thrs2', 'edge', 4000, 5000, nothing)

    cap = cv.VideoCapture("../20230605_163120.mp4")

    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    end = False

    curr_frame = 0
    while curr_frame < int(frame_count) and not end:
        for i in range(7):
            cap.read()

        _flag, img = cap.read()

        img = cv.resize(img, None, fx=0.25, fy=0.25)

        # Grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        blur = cv.GaussianBlur(gray, (5, 5), 0)

        # Canny Detection
        edge = cv.Canny(blur, 2000, 4000, apertureSize=5)

        mask_white = cv.inRange(blur, 200, 240)
        mask_white_image = cv.bitwise_and(blur, mask_white)

        vis = img.copy()
        vis = np.uint8(vis/2.)
        vis[edge != 0] = (0, 255, 0)

        vis2 = vis.copy()
        vis2 = np.uint8(vis2 / 2.)
        vis2[mask_white_image != 0] = (0, 0, 255)

        cv.imshow('edge', vis2)

        # vis3 = mask_white_image.copy
        # vis3 = np.uint8(vis3 / 2.)
        # vis3[edge != 0] = (0, 0, 255)

        # cv.imshow('edge', vis3)
        key = cv.waitKey(1)
        if key == 27:
            break
        curr_frame += 1

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
