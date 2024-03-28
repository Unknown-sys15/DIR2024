from camera1 import Vision
import cv2
import sys
import numpy as np

# Create a Vision object
vision = Vision()

# Load the captured frame
frame = cv2.imread("image.png")

frame = frame[150:250, 200:300]

# Zoom image
frame = cv2.resize(frame, (0, 0), fx=2, fy=2)

# Display the frame
cv2.imshow("Frame", frame)
cv2.waitKey(0)

# Find rectangles in the frame
picture, rectangles = vision.get_rectangles(frame, cutoff=500)

# Display the frame with rectangles
cv2.imshow("Frame", picture)
cv2.waitKey(0)

def zeroPad(image, m, n):
    # Your Code Here
    X, Y = image.shape
    padded_image = np.zeros((X+2*m, Y+2*n))
    padded_image[m:X+m, n:Y+n] = image
    return padded_image

# Geometric rectification
if len(rectangles) == 1:
    rectified = vision.geometric_rectification(frame, rectangles[0])
    x = rectified
    # Display the rectified frame
    cv2.imshow("Rectified", rectified)
    cv2.waitKey(0)
    # Canny image
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    # Improve edge detection
    gray = cv2.GaussianBlur(gray, (5, 5), 0.5)
    edges = cv2.Canny(gray, 200, 300)
    # Display the edges
    cv2.imshow("Edges", edges)
    cv2.waitKey(0)


    # Rotate the frame 90
    frame = cv2.rotate(edges, cv2.ROTATE_90_CLOCKWISE)

    print("Frame size: ", frame.shape)

    cv2.imshow("Frame", frame)
    cv2.waitKey(0)

    # Orientation deteciton
    #orientation, orient_frame = vision.anomaly_detection(frame, rectangles[0])

print("Orientation: ", orientation)

# Display the frame with orientation
cv2.imshow("Orientation", orient_frame)
cv2.waitKey(0)
