# External module imports.
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import plotly.express as px

# Internal module imports.
from essential_matrix_computation import essential_matrix_computation
from featuretracking import feature_tracking
from corner_detection_fast import extract_features

videopath = "C:/Users/tanus/Downloads/20230605_163120_Trim2.mp4"
cap = cv2.VideoCapture(videopath)
if not cap.isOpened():
    print("Access error")
    exit()

idx = 0
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Store one frame behinds info
prevFrameCorners = []
prevFrame = ()
rotationArr = np.ones(shape=(3, 3))
translationArr = np.empty(shape=(3, 1))
focal = 718.8560
pp = (607.1928, 185.2157)

current_point = np.array([1, 1, 1])

path = [current_point]
colours = [(0, 0, 0)]
step = 1

for i in tqdm(range(int(frame_count))):
    ret, frame = cap.read()
    if not ret:
        print("Unable to read the frame")
        continue

    # Always do extraction
    currentFrameCorners = extract_features(frame, True)

    if i != 0:
        # logic with prevFrameCorners which would be i-1 frame, and currentFrameCorners
        prevCorners, curCorners = feature_tracking(
            prevFrame, frame, prevFrameCorners, currentFrameCorners
        )
        rotationArr, translationArr = essential_matrix_computation(
            rotationArr,
            translationArr,
            curCorners,
            prevCorners,
            focal,
            pp,
            i,
            None,
            True
        )

        # Matrixes updated, and corners updated
        # plot result of our matrix change

    prevFrameCorners = currentFrameCorners  # Save i-1 frame
    prevFrame = frame
    # cv2.imwrite(f"frame_{idx}.jpg", frame)
    idx += 1

    # Plotting points after rotation and translation.
    rotated_point = np.matmul(rotationArr, current_point).reshape(3,1)


    translated_point = np.multiply(translationArr, rotated_point)
    current_point = translated_point
    path.append(current_point)

    # Distinguishing points in time by colour gradient.
    new_colour = (cmp + step for cmp in colours[idx - 1])
    colours.append(new_colour)

path = path[1:-1]
x = [point[0] for point in path]
y = [point[1] for point in path]
z = [point[2] for point in path]

flattened_x = np.array([item for sublist in x for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])])
flattened_y = np.array([item for sublist in y for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])])

# Plotting points with matplotlib in 2D.
plt.title("Robot Path")
plt.plot(flattened_x, flattened_y)
plt.show()

# Plotting points with plotly in 3D.
plot = px.scatter_3d(x=x, y=y, z=z, color=colours)
plot.show()

cap.release()
