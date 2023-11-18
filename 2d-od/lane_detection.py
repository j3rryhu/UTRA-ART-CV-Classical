
'''
This sample demonstrates line detection with Hough Transform.

'''

import cv2 as cv
import numpy as np

def main():
    cv.namedWindow('Hough Lines')

    cap = cv.VideoCapture("../assets/20230605_163120.mp4")

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

        # Mask for white areas
        mask_white = cv.inRange(blur, 200, 240)
        mask_white_image = cv.bitwise_and(blur, mask_white)

        # Detect edges for Hough transformation
        edges = cv.Canny(mask_white_image, 50, 150, apertureSize=3)

        # Hough Line Transformation
        lines = cv.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=40, maxLineGap=10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv.imshow('Hough Lines', img)

        key = cv.waitKey(1)
        if key == 27:
            break
        curr_frame += 1

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
