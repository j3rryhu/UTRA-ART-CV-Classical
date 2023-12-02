import numpy as np
import cv2
from matplotlib import pyplot as plt
from common_functions import *
from image_processing import process_image


def extract_features(image, isNmsEnabled: bool) -> any:
    # Read image in grayscale.
    # image = cv2.imread("cuboid.jpeg", cv2.IMREAD_GRAYSCALE)

    # Instantiate FAST detector.
    fast_feature_detector = cv2.FastFeatureDetector_create()

    if isNmsEnabled:
        # Get corners.
        corners = fast_feature_detector.detect(image, None)
    else:
        fast_feature_detector.setNonmaxSuppression(0)
        corners = fast_feature_detector.detect(image, None)

    return corners


def show_image_with_corners(image):
    corners_with_nms = extract_features(image, False)
    corners_without_nms = extract_features(image, True)

    cornered_image_with_nms = cv2.drawKeypoints(
        image, corners_with_nms, None, color=(255, 0, 0)
    )
    cornered_image_without_nms = cv2.drawKeypoints(
        image, corners_without_nms, None, color=(255, 0, 0)
    )
    show_image("cornered_image_with_nms", cornered_image_with_nms)
    show_image("cornered_image_without_nms", cornered_image_without_nms)


if __name__ == "__main__":
    image = cv2.imread("frames/frame_0.jpg")
    image = process_image(image)

    show_image("Processed image", image)
    show_image_with_corners(image)
