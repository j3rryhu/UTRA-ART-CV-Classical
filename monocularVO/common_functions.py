import cv2


def show_image(window_name, image):
    cv2.imshow(window_name, image)
    if cv2.waitKey(0) == "q":
        cv2.destroyWindow(window_name)
