import cv2


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

