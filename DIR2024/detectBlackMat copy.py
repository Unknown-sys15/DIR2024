import cv2
import numpy as np

def empty():
    pass

def crop_image_with_four_points(image, points):
    """
    Crop an image using four points defining a quadrilateral region.

    Args:
        image (numpy.ndarray): Input image.
        points (list of tuples): Four points defining a quadrilateral region to crop.
                                  Each point should be a tuple (x, y) representing the coordinates.

    Returns:
        numpy.ndarray: Cropped image.
    """
    # Define the width and height of the output cropped image
    width = int(max(np.sqrt((points[0][0] - points[1][0]) ** 2 + (points[0][1] - points[1][1]) ** 2),
                    np.sqrt((points[2][0] - points[3][0]) ** 2 + (points[2][1] - points[3][1]) ** 2)))
    height = int(max(np.sqrt((points[1][0] - points[2][0]) ** 2 + (points[1][1] - points[2][1]) ** 2),
                     np.sqrt((points[3][0] - points[0][0]) ** 2 + (points[3][1] - points[0][1]) ** 2)))

    # Define the four points of the output rectangle
    pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    # Convert input points to numpy array
    pts_src = np.array(points, dtype=np.float32)

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # Apply the perspective transformation to the image
    cropped_image = cv2.warpPerspective(image, M, (width, height))

    return cropped_image

# Load the image
image = cv2.imread('black.jpg')
image = image[50:380, 250:520]
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,940)
cv2.createTrackbar("x1","Parameters",10,255,empty)
cv2.createTrackbar("y1","Parameters",10,255,empty)

cv2.createTrackbar("x2","Parameters",200,255,empty)
cv2.createTrackbar("y2","Parameters",10,255,empty)

cv2.createTrackbar("x3","Parameters",600,900,empty)
cv2.createTrackbar("y3","Parameters",10,900,empty)

cv2.createTrackbar("x4","Parameters",600,900,empty)
cv2.createTrackbar("y4","Parameters",600,900,empty)


while True:
# Define the four points of the region to crop
    x1 = cv2.getTrackbarPos("x1", "Parameters")
    y1 = cv2.getTrackbarPos("y1", "Parameters")
    x2 = cv2.getTrackbarPos("x2", "Parameters")
    y2 = cv2.getTrackbarPos("y2", "Parameters")
    x3 = cv2.getTrackbarPos("x3", "Parameters")
    y3 = cv2.getTrackbarPos("y3", "Parameters")
    x4 = cv2.getTrackbarPos("x4", "Parameters")
    y4 = cv2.getTrackbarPos("y4", "Parameters")
    
    pts_src = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)

    # Define the width and height of the output cropped image
    width, height = 500, 500

    # Define the four points of the output rectangle
    pts_dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)

    # Apply the perspective transformation to the image
    cropped_image = cv2.warpPerspective(image, M, (width, height))

    # Display the original and cropped images
    cv2.imshow('Original Image', crop_image_with_four_points(image ,[[x1,y1], [x2, y2], [x3, y3], [x4, y4]]))
    cv2.imshow('Cropped Image', cropped_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()