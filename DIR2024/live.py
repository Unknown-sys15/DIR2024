import cv2

def main():
    # Create a VideoCapture object
    cap = cv2.VideoCapture(1)  # Use 0 for the default camera, or specify the camera index if you have multiple cameras

    if not cap.isOpened():
        print("Error: Failed to open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Check if the frame was successfully captured
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Display the frame
        cv2.imshow('Video', frame)

        # Check for the 'q' key pressed to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()