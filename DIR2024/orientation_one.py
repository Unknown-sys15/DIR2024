from camera import Camera
import cv2
import numpy as np

def show_image(frame):
    # Display the captured frame
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)

# Create a Camera object
print("initializing camera...")
#camera = Camera()


# Capture a frame from the camera
print("Capturing frame...")
#frame = camera.capture_and_save("image.jpg")
print("Frame captured")


# Load the captured frame
frame = cv2.imread("C:\\Users\\janra\Desktop\\DIR2024\\test_one.jpg")
print(frame)
cv2.imshow("frame",frame)
cv2.waitKey(0)


# Detect edges in the frame
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 250, 300)

# Display the edges
show_image(edges)

# find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# approximate the contours with rectangles
rectangles = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > 0]

# sort rectangles by area
rectangles = sorted(rectangles, key=lambda rect: rect[2] * rect[3], reverse=True)

# Draw the rectangles on the frame
for x, y, w, h in rectangles:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Draw center of rectangles
for x, y, w, h in rectangles:
    cv2.circle(frame, (x + w // 2, y + h // 2), 5, (0, 0, 255), -1)

centers = [(x + w // 2, y + h // 2) for x, y, w, h in rectangles if len(rectangles) == 2]

# Draw vector   
cv2.line(frame, centers[0], centers[1], (255, 0, 0), 2)


# Display the frame with rectangles
show_image(frame)

print(frame.shape)


 