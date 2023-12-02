import cv2
from common_functions import *
from input_reading import ZedCamera

def blur(image):
    return cv2.bilateralFilter(image, 9, 175, 175)
    # return cv2.GaussianBlur(image, (11, 11), 0)


def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def undistort(image, zedCamera: ZedCamera):
    return cv2.undistort(image, zedCamera.camera_matrix, zedCamera.dist, None, zedCamera.new_camera_mtx)

def process_image(image, functions=[undistort,gray, blur]):
    for function in functions:
        image = function(image)
    return image


if __name__ == "__main__":
    image = cv2.imread("frames/frame_0.jpg")
    processed_image = process_image(image)
    show_image("Processed Image", processed_image)
