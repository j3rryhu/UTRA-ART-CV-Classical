# External module imports.
import cv2
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import plotly.express as px
from skimage.io import imread_collection

# Internal module imports.
from essential_matrix_computation import essential_matrix_computation
from featuretracking import feature_tracking
from corner_detection_fast import extract_features
from image_processing import process_image
from common_functions import *

rotate_180 = np.matrix([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

idx = 0

# Store previous frame's info
prevFrameCorners = []
prevFrame = ()
rotationArr = np.eye(3)
# rotationArr = rotate_180
translationArr = np.empty(shape=(3, 1))

camera_matrix = np.matrix([[2035.62, 0, 773.202], [0, 2019.24, 1360.42], [0, 0, 1]])
focal = 90.7
pp = (773.202, 1360.42)

current_point = np.array([0, 0, 0])

path = [current_point]
colours = [(0, 0, 0)]
step = 1

dir_name = "frames/*.jpg"

# Read all images.
images = imread_collection(dir_name)


def plot_path(path):
    path = path[1:-1]
    x = [point[0][0] for point in path]
    y = [point[1][0] for point in path]
    z = [point[2][0] for point in path]

    flattened_x = np.array(
        [
            item
            for sublist in x
            for item in (
                sublist if isinstance(sublist, (list, np.ndarray)) else [sublist]
            )
        ]
    )
    flattened_y = np.array(
        [
            item
            for sublist in y
            for item in (
                sublist if isinstance(sublist, (list, np.ndarray)) else [sublist]
            )
        ]
    )

    # Plotting points with matplotlib in 2D.
    plt.title("Robot Path")
    plt.plot(flattened_x, flattened_y)
    plt.show()


for i in range(len(images)):
    # Process image
    frame = process_image(images[i])

    # Always do extraction
    currentFrameCorners = extract_features(frame, True)
    # print(len(currentFrameCorners))

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
            True,
            camera_matrix,
        )

        # rotationArr = np.matmul(rotate_180, rotationArr)

        # print("Rotation Matrix ", rotationArr)
        # print("Translation Matrix", translationArr)

        # Matrixes updated, and corners updated
        # plot result of our matrix change

    prevFrameCorners = currentFrameCorners  # Save i-1 frame
    prevFrame = frame
    # cv2.imwrite(f"frame_{idx}.jpg", frame)
    idx += 1

    # Plotting points after rotation and translation.
    rotated_point = np.matmul(rotationArr, current_point).reshape(3, 1)

    translated_point = rotated_point + translationArr
    current_point = translated_point
    path.append(current_point)

    # Distinguishing points in time by colour gradient.
    new_colour = (cmp + step for cmp in colours[idx - 1])
    colours.append(new_colour)

    # plot_path(path)

path = path[1:-1]

x = [point[0][0] for point in path]

y = [-point[1][0] for point in path]
z = [point[2][0] for point in path]


flattened_x = np.array(
    [
        item
        for sublist in x
        for item in (
            sublist if isinstance(sublist, (list, np.ndarray, np.matrix)) else [sublist]
        )
    ]
)

flattened_y = np.array(
    [
        item
        for sublist in y
        for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])
    ]
)

# Plotting points with matplotlib in 2D.
plt.title("Robot Path")
plt.plot(flattened_x, flattened_y)
plt.show()

# Plotting points with plotly in 3D.
# plot = px.scatter_3d(x=x, y=y, z=z, color_continuous_scale="Viridis")
# plot.show()

plot = px.scatter_3d(
    x=x,
    y=y,
    z=np.arange(len(path)),
    color_continuous_scale="Viridis",
)
# plot.update_layout(updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play',
#                                             method='animate', args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)])])])
plot.update_traces(marker_size=3)
plot.show()
cap.release()
