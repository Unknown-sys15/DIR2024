import cv2
import numpy as np


class Camera:
    def __init__(self):
        # Open the camera
        self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

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
        return frame

class Vision:
    def __init__(self):
        # Create rectif rectangle for vision 1
        frame = cv2.imread("calibrate.jpg")
        print(frame.shape)
        frame = frame[30:370,100:420]


        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_black = np.array([180,98,80])
        upper_black = np.array([0,0,0])
        mask = cv2.inRange(hsv,upper_black,lower_black)

        #cv2.imshow("Mask", mask)

        mask = cv2.GaussianBlur(mask, (9, 9), 0.5)
        edges = cv2.Canny(mask, 600, 600)
        # Define a kernel for dilation
        kernel = np.ones((3, 3), np.uint8)/3
        # Perform dilation to close contours
        edges = cv2.dilate(edges, kernel, iterations=1)

        # find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # fit rectangles to the contours
        rectangles = [cv2.minAreaRect(contour) for contour in contours if cv2.contourArea(contour) > 500]

        draw_on = np.copy(frame)
        # Draw the rectangles on the frame

        for rectangle in rectangles:
            points = cv2.boxPoints(rectangle)
            points = np.int0(points)
            for point in points:
                point[0] -= 0
                point[1] -= -5

            # Draw the rectangle
            cv2.drawContours(draw_on, [points], 0, (0, 255, 0), 1)
        
        #cv2.imshow("Main rectangle", draw_on)
        #cv2.waitKey(0)


        
        if len(rectangles) != 1:
            print("Wrong number of rectangles found, calibration failed")

            cv2.imshow("maska", mask)
            cv2.imshow("Edges", edges)
            cv2.imshow("Main rectangle", draw_on)
            cv2.waitKey(0)


        
        self.rect = rectangles[0]

    def get_rectangles(self,frame, thr1 = 150, thr2 = 200, gaussian_blur=False, dilate = (5, 5), cutoff = 50, cutoff2=0):
        # Detect edges in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Improve edge detection
        if gaussian_blur:
            gray = cv2.GaussianBlur(gray, gaussian_blur[0:2], gaussian_blur[2])
        edges = cv2.Canny(gray, thr1, thr2)
        if dilate:
            kernel = np.ones(dilate, np.uint8)
            closed_edges = cv2.dilate(edges, kernel, iterations=1)
        # find contours
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # fit rectangles to the contours
        rectangles = [cv2.minAreaRect(contour) for contour in contours if cv2.contourArea(contour) > cutoff]

        rectangles = [rectangle for rectangle in rectangles if rectangle[1][0]*rectangle[1][1] > cutoff2]


        draw_on = np.copy(frame)
        # Draw the rectangles on the frame
        
        for rectangle in rectangles:
            points = cv2.boxPoints(rectangle)
            points = np.int0(points)
            # Draw the rectangle
            cv2.drawContours(draw_on, [points], 0, (0, 255, 0), 1)
        
        return frame, draw_on, rectangles
    
    
    def found(self, rectangles):
        if len(rectangles) == 1:
            rectangle = rectangles[0]
            if rectangle[1][0] < 0 or rectangle[1][1] < 0 or rectangle[2] > 0:
                print("Rectangle not found")
                return False
            else:
                return True
        else:
            return True
    
    def anomaly_orientation_preprocess(self, frame, rectangle):        
        rectified = self.geometric_rectification(frame, rectangle)

        # Display the rectified frame
        #cv2.imshow("Rectified", rectified)
        # Canny image
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        # Improve edge detection
        gray = cv2.GaussianBlur(gray, (5, 5), 0.5)
        edges = cv2.Canny(gray, 200, 300)
        # Define a kernel for dilation
        kernel = np.ones((3, 3), np.uint8)
        # Perform dilation to close contours
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Display the edges
        cv2.imshow("Edges", edges)

        return edges

    def orientation_detection(self, edges):

        # calculate mean of rectangles for orientation detection
        mean1 = np.mean(edges[-15:,:])
        mean2 = np.mean(edges[:15,:])

        if mean1 > mean2:
            orientation = 1
        else:
            orientation = 0

        return orientation
    
    def anomaly_detection(self, edges):
        # calculate mean of rectangles for anomaly deteciton

        mean1 = np.mean(edges[:,:15])
        mean2 = np.mean(edges[:,-15:])

        threshold = 50
        if (mean1 > threshold) and (mean2 > threshold):
            anomaly = False
        else:
            anomaly = True

        return anomaly
    
    def pallete(self, x,y, x_max, y_max):

        T1 = np.array([353.85,-167.12,15.25])
        T2 = np.array([206.24,-165.94,15.25])
        T3 = np.array([203.97,-374.12,15.25])

        vx = T1 - T2
        vy = T2 - T3

        return T3 + (y/y_max)*vx + (x/x_max)*vy
    
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
    
    def vision1(self, frame):
        frame = frame[30:370,100:420]

        # geometric rectification
        rectified = self.geometric_rectification(frame, self.rect)
        #cv2.imshow("rectified", rectified)
        
        #rectified = cv2.rotate(rectified, cv2.ROTATE_90_CLOCKWISE)
        rectified = rectified[2:-8,5:-2]

        #cv2.imshow("rectified2", rectified)
        


        rectified, picture, rectangles = self.get_rectangles(rectified, thr1 = 200, thr2 = 300, gaussian_blur=(5, 5, 0.5), dilate=(3, 3), cutoff=350)


        if not self.found(rectangles):
            print("Rectangle not found")
            cv2.imshow("Rectified", rectified)
            cv2.imshow("Rectified", rectified)

            cv2.waitKey(0)

            return -1
        
        # Draw center of rectangles
        for rectangle in rectangles:
            center = np.int0(rectangle[0])
            cv2.circle(picture, tuple(center), 3, (0, 0, 255), -1)

        # Show the frame with rectangles
        cv2.imshow("Rectangle", picture)

        # Sort rectangles by area
        rectangles = sorted(rectangles, key=lambda x: x[1][0] * x[1][1], reverse=True)

        # Convert from tuples to list:
        points = []
        for rectangle in rectangles:
            points.append([rectangle[0][0], rectangle[0][1], rectangle[2]])

        # convert to true orientation
        for rectangle,point in zip(rectangles,points):
            bbox = np.array(cv2.boxPoints(rectangle))
            if np.linalg.norm(bbox[0] - bbox[3]) < np.linalg.norm(bbox[0] - bbox[1]):
                point[2] = point[2] - 90
            point[2] = -point[2]

        #center = np.int0(rectangle[0])
        #angle = rectangle[2]
        
        transformed_points = []
        for p in points:
            transformed_points.append([self.pallete(p[0],p[1],rectified.shape[1],rectified.shape[0]), p[2]])
            
        return transformed_points
    
    def vision2(self, frame):

        #cv2.imshow("Frame", frame)
        frame = frame[100:200,430:600]

        #cv2.imshow("Frame2", frame)
        # Zoom image
        frame = cv2.resize(frame, (0, 0), fx=2, fy=2)
        # Display the frame
        #cv2.imshow("Crop + zoom", frame)

        # Find rectangles in the frame
        frame, picture, rectangles = self.get_rectangles(frame, thr1 = 80, thr2 = 200, gaussian_blur=(7,7,0.5), cutoff=100)



        if not self.found(rectangles):
            print("Rectangle not found")
            return -1, -1

        # Sort rectangles by area
        rectangles = sorted(rectangles, key=lambda x: x[1][0] * x[1][1], reverse=True)
        rectangles = [rectangles[0]]

        # Display the frame with rectangles
        #cv2.imshow("Rectangles", picture)
        if len(rectangles) == 1:
            edges = self.anomaly_orientation_preprocess(frame, rectangles[0])
            orientation = self.orientation_detection(edges)
            anomaly = self.anomaly_detection(edges)

            return orientation, anomaly
        else:
            print("Wrong number of rectangles found")
            return -1, -1





