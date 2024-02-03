import cv2


def extract_features(image, isNmsEnabled: bool) -> any:
    # Initiate ORB detector.

    # Number of features retained.
    num_features = 100

    # Use FAST_SCORE to sacrifice repeatability for speed.
    orb = cv2.ORB_create(nfeatures=num_features, scoreType=cv2.HARRIS_SCORE)

    # Amount of the borders to ignore.
    edgeThreshold = 20
    orb.setEdgeThreshold(edgeThreshold)

    if isNmsEnabled:
        # Get corners.
        corners = orb.detect(image, None)

    else:
        orb.setNonmaxSuppression(0)
        corners = orb.detect(image, None)

    return corners
