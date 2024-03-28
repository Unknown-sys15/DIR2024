from camera import Vision, Camera
import cv2
import sys
import numpy as np

from scipy.spatial.transform import Rotation

# Function to convert Euler angles to quaternions
def euler_to_quaternion(angles):
    # Convert Euler angles to rotation matrix
    rotation_matrix = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
    
    # Convert rotation matrix to quaternion
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
    
    return quaternion

def xy_angle_toString(data):
    angle = data[1]
    angle += 180
    if angle > 180:
        angle -360 
    data = np.array(data[0])
    xy = data[:2]
    xy = np.round(xy, 3)
    quaternion = euler_to_quaternion([-180,0,angle])
    quaternion = np.array([quaternion[2], quaternion[0], quaternion[1], quaternion[3]])
    quaternion = np.round(quaternion, 3)

    return ';'.join(xy.astype(str))+';',';'.join(quaternion.astype(str))+';' 

def create_moveString(data, nad):
    if nad == True:
        z = str(10.84 + 15) + ';'
    else:
        z = str(10.84) + ';'
    return xy_angle_toString(data)[0] + z + xy_angle_toString(data)[1] + '-1;0;-1;0;'



# Create a Vision object






def do_vision(vision):
    print("Doing vision...")
    points = -1
    orient = -1
    anomaly = -1
    vision_obj = Vision()

    camera = Camera()
    
    if vision == 1:
        moves = []
        print("Detecting objects...")
        while points == -1:
            frame = camera.capture_and_save("last_taken_photo.jpg")
            points = vision_obj.vision1(frame)
            print("Trying...")

        for point in points:
            moves.append(create_moveString(point, nad=True))
            moves.append(create_moveString(point, nad=False))

        print("Success: Press 0...")
        cv2.waitKey(0)

        
        return moves
    
    if vision == 2:
        print("Quality checking...")
        while orient and anomaly == -1:
            frame = camera.capture_and_save("last_taken_photo.jpg")
            orient, anomaly = vision.vision2(frame)
            print("Trying...")
        
        print(f"Success: orientation = {orient}, anomaly = {anomaly}")
        return orient, anomaly




#print(do_vision(vision=1))


# # VISION 1
# while points == -1:
#     frame = camera.capture_and_save("last_taken_photo.jpg")
#     points = vision.vision1(frame)

# coords = xy_angle_toString(points)
# print(coords)

# #frame = cv2.imread("an0.png")

# # VISION 2
# while orient and anomaly == -1:
#     frame = camera.capture_and_save("last_taken_photo.jpg")
#     orient, anomaly = vision.vision2(frame)









