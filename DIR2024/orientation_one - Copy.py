from camera import Camera
import cv2
import numpy as np

def show_image(frame):
    # Display the captured frame
    cv2.imshow("Frame", frame)
    cv2.waitKey(0)
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def empty():
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",61,255,empty)
cv2.createTrackbar("Threshold2","Parameters",108,255,empty)
cv2.createTrackbar("Threshold3","Parameters",108,1500,empty)

# Create a Camera object
print("initializing camera...")
#camera = Camera()


# Capture a frame from the camera
print("Capturing frame...")
#frame = camera.capture_and_save("image.jpg")


#cap = cv2.VideoCapture(1)
#success, frame = cap.read()
#cv2.imwrite("C:\\Users\\janra\Desktop\\DIR2024\\testnna_slikica1234.jpg", frame)
print("Frame captured")

#if not cap.isOpened():
#    print("err")
#else:
#    pass

frame = cv2.imread("C:\\Users\\janra\Desktop\\DIR2024\\testnna_slikica1234.jpg")

def getContours(img,imgContour,ContourVelicina):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)#CHAIN_APPROX_SIMPLE
    #cv2.drawContours(imgContour,contours,-1,(255, 0, 255),7)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area>ContourVelicina:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt,0.02*peri,True)#ugibas shape conture
            print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)#izrisi bounding rect

            if objCor ==3: objectType ="Tri"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio >0.98 and aspRatio <1.03: objectType= "Square"
                else:objectType="Rectangle"
            elif objCor>4: objectType= "OSTALO"
            else:objectType="None"



            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imgContour,objectType,
                        (x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                        (0,0,0),2)

while True:
# Load the captured frame
    
    #success, frame = cap.read()
    #print(frame)
#cv2.imshow("frame",frame)
#cv2.waitKey(0)
    imgContour = frame.copy()

    imgBlur = cv2.GaussianBlur(frame, (7,7), 1) #kernel 7 by 7
    gray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
#cv2.imshow("frame",frame)
#cv2.waitKey(0)

    Threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    ContourVelicina = cv2.getTrackbarPos("Threshold3", "Parameters")
    imgCanny = cv2.Canny(gray,Threshold1,Threshold2)

    kernel = np.ones((5,5))
    imgDil = cv2.dilate(imgCanny,kernel,iterations=1)

    getContours(imgDil,imgContour,ContourVelicina)



    imgStack = stackImages(0.8,([frame,imgCanny],[imgDil,imgContour]))
    cv2.imshow("res",imgStack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




'''


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


 '''