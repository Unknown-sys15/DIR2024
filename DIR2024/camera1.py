import cv2
import numpy as np


class Camera:
    def __init__(self):
        # Open the camera
        self.cap = cv2.VideoCapture(1)

        # Check if the camera is opened successfully
        if not self.cap.isOpened():
            print("Unable to open the camera")
            exit()
    
    def capture(self):
        # Capture a frame from the camera
        ret, frame = self.cap.read()

        # Check if the frame was captured successfully
        if not ret:
            print("Failed to capture frame")
            exit()

        return frame
    
    def release(self):
        # Release the camera
        self.cap.release()

    def capture_and_save(self, filename):
        # Capture a frame from the camera
        frame = self.capture()

        # Save the captured frame to a file
        cv2.imwrite(filename, frame)

class Vision:
    def __init__(self):
        pass

    def multiple_detection(self, frame):
        # Detect edges in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Improve edge detection
        gray = cv2.GaussianBlur(gray, (5, 5), 0.5)
        edges = cv2.Canny(gray, 200, 300)
        # Define a kernel for dilation
        kernel = np.ones((3, 3), np.uint8)
        # Perform dilation to close contours
        closed_edges = cv2.dilate(edges, kernel, iterations=1)
        # find contours
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # fit rectangles to the contours
        rectangles = [cv2.minAreaRect(contour) for contour in contours if cv2.contourArea(contour) > 0]
        # Draw the rectangles on the frame
        for rectangle in rectangles:
            points = cv2.boxPoints(rectangle)
            points = np.int0(points)
            # Draw the rectangle
            cv2.drawContours(frame, [points], 0, (0, 255, 0), 1)
        # Draw center of rectangles
        for rectangle in rectangles:
            center = np.int0(rectangle[0])
            cv2.circle(frame, tuple(center), 3, (0, 0, 255), -1)
        # extract postition and orientation of rectangles
        for rectangle in rectangles:
            center = np.int0(rectangle[0])
            angle = rectangle[2]
        return rectangles, frame
    
    def calculate_mean_of_rectangle(self, frame, rectangle):
        x, y, width, height = rectangle
        print("rectangle in mean", rectangle)
        # Calculate the mean of the rectangle
        roi = frame[y:y+height, x:x+width]
        mean = np.mean(roi)
        return mean
    
    def get_rectangles(self,frame, cutoff=50):
        # Detect edges in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Improve edge detection
        #gray = cv2.GaussianBlur(gray, (7, 7), 1)
        edges = cv2.Canny(gray, 150, 200)
        kernel = np.ones((5, 5), np.uint8)
        closed_edges = cv2.dilate(edges, kernel, iterations=1)
        # find contours
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # fit rectangles to the contours
        rectangles = [cv2.minAreaRect(contour) for contour in contours if cv2.contourArea(contour) > cutoff]

        draw_on = frame
        # Draw the rectangles on the frame
        for rectangle in rectangles:
            points = cv2.boxPoints(rectangle)
            points = np.int0(points)
            # Draw the rectangle
            cv2.drawContours(draw_on, [points], 0, (0, 255, 0), 1)

        return draw_on, rectangles

    
    def draw_rectangle(self, frame, rectangle):
        x, y, width, height = rectangle
        x = x - width // 2
        y = y - height // 2
        # Draw the rectangle
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 0, 0), 2)
        return frame

    def orientation_detection(self, frame):
        # Detect edges in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Improve edge detection
        gray = cv2.GaussianBlur(gray, (9, 9), 1)
        edges = cv2.Canny(gray, 68, 51)
        kernel = np.ones((5, 5), np.uint8)
        closed_edges = cv2.dilate(edges, kernel, iterations=1)
        # find contours
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # fit rectangles to the contours
        rectangles = [cv2.minAreaRect(contour) for contour in contours if cv2.contourArea(contour) > 50]
        if len(rectangles) == 1:
            rectangle = rectangles[0]

            w1 = 30
            h1 = 60

            w2 = 30
            h2 = 60
            # define orientation rectangle
            orientation_rectangle1 = (np.int0(rectangle[0][0]),np.int0(rectangle[0][1]+rectangle[1][0]/2),w1,h1)
            orientation_rectangle2 = (np.int0(rectangle[0][0]),np.int0(rectangle[0][1]-rectangle[1][0]/2),w2,h2)
            orientation_rectangles = [orientation_rectangle1, orientation_rectangle2]
        else:
            print("Could not detect orientation", len(rectangles))
            return -1, closed_edges

        # calculate mean of rectangles for orientation detection
        mean1 = self.calculate_mean_of_rectangle(closed_edges, orientation_rectangle1)
        mean2 = self.calculate_mean_of_rectangle(closed_edges, orientation_rectangle2)

        print("Mean1, Mean2")
        print(mean1, mean2)


        if mean1 > mean2:
            orientation = 1
        else:
            orientation = 0

        draw_on = frame
        for rect in orientation_rectangles:
            draw_on = self.draw_rectangle(draw_on, rect)

        return orientation, draw_on
    
    def anomaly_detection(self, frame, rectangle):
        w1 = 10
        h1 = 10

        w2 = 10
        h2 = 10

        w3 = 10
        h3 = 10

        print("Rectangle: ", rectangle)
        # define error correction rectangles
        #error_correction_rectangle1 = (np.int0(rectangle[0][0]),np.int0(rectangle[0][1]-rectangle[1][0]/2+h1/2),w1,h1)
        #error_correction_rectangle2 = (np.int0(rectangle[0][0]-rectangle[1][1]/2 - w2/2),np.int0(rectangle[0][1]-rectangle[1][0]/2+h2/2),w2,h2)
        #error_correction_rectangle3 = (np.int0(rectangle[0][0]+rectangle[1][1]/2 + w3/2),np.int0(rectangle[0][1]-rectangle[1][0]/2+h3/2),w3,h3)
        #error_correction_rectangles = [error_correction_rectangle1, error_correction_rectangle2, error_correction_rectangle3]

        # calculate mean of rectangles for anomaly deteciton
        mean = 0
        for rect in error_correction_rectangles:
            mean += self.calculate_mean_of_rectangle(frame, rect)
        mean /= len(error_correction_rectangles)

        threshold = 10 # define threshold?
        if mean > threshold:
            anomaly = True
        else:
            anomaly = False

        draw_on = frame
        for rect in error_correction_rectangles:
            draw_on = self.draw_rectangle(draw_on, rect)

        return anomaly, draw_on
    
    def geometric_rectification(self, frame, rectangle):
        # Get the points of the rectangle
        points = cv2.boxPoints(rectangle)
        points = np.int0(points)
        # Get the width and height of the rectangle
        width = int(rectangle[1][0])
        height = int(rectangle[1][1])
        # Define the destination points
        dst_points = np.array([[0, 0], [height, 0], [height, width], [0, width]], np.float32)
        # Get the transformation matrix
        M = cv2.getPerspectiveTransform(np.float32(points), dst_points)
        # Perform the transformation
        rectified = cv2.warpPerspective(frame, M, (height, width))
        return rectified
