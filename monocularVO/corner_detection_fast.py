import numpy as np
import cv2
from matplotlib import pyplot as plt
import common_functions


def extract_features(image, isNmsEnabled: bool) -> any:
    # Read image in grayscale.
    #image = cv2.imread("cuboid.jpeg", cv2.IMREAD_GRAYSCALE)

    # Instantiate FAST detector.
    fast_feature_detector = cv2.FastFeatureDetector_create()

    if isNmsEnabled:
        # Get corners.
        corners = fast_feature_detector.detect(image, None)
    else:
        fast_feature_detector.setNonmaxSuppression(0)
        corners = fast_feature_detector.detect(image, None)

    return corners


def show_image_with_corners(image_name: str) -> any:
    image = cv2.imread(image_name)
    corners_with_nms = extract_features(image, False)
    corners_without_nms = extract_features(image, True)

    cornered_image_with_nms = cv2.drawKeypoints(
        image, corners_with_nms, None, color=(255, 0, 0)
    )
    cornered_image_without_nms = cv2.drawKeypoints(
        image, corners_without_nms, None, color=(255, 0, 0)
    )
    common_functions.show_image("cornered_image_with_nms", cornered_image_with_nms)
    common_functions.show_image(
        "cornered_image_without_nms", cornered_image_without_nms
    )


if __name__ == "__main__":
    show_image_with_corners("cuboid.jpeg")
