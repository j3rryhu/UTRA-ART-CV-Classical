import numpy as np
import cv2
import os


def feature_tracking(image1, image2, corners1, corners2):
    corners1 = np.array([kp.pt for kp in corners1], dtype=np.float32).reshape(-1, 1, 2)
    corners2 = np.array([kp.pt for kp in corners2], dtype=np.float32).reshape(-1, 1, 2)

    corners2, status, err = cv2.calcOpticalFlowPyrLK(
        image1,
        image2,
        corners1,
        None,
        **dict(
            winSize=(21, 21),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
    )
    # To get rid of points for the KLT tracking where the status is 0 or if they are not in the frame

    corners1 = corners1[status == 1]
    corners2 = corners2[status == 1]

    return corners1, corners2
