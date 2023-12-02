import cv2
from tqdm import tqdm


def show_image(window_name, image):
    cv2.imshow(window_name, image)
    if cv2.waitKey(0) == "q":
        cv2.destroyWindow(window_name)


def get_frames(video_name, start_frame, num_frames, step):
    cap = cv2.VideoCapture(video_name)

    if not cap.isOpened():
        print("Access error")
        exit()

    idx = start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    while idx < start_frame + num_frames:
        ret, frame = cap.read()
        if not ret:
            print("Unable to read the frame")
            continue

        cv2.imwrite(f"frames/frame_{idx}.jpg", frame)
        idx += step
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    cap.release()


if __name__ == "__main__":
    get_frames("competition_data.mp4", 1000, 200, 5)
