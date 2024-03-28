from camera import Camera
import cv2
import numpy as np

def empty():
    pass
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



cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",212,255,empty)
cv2.createTrackbar("Threshold2","Parameters",143,255,empty)
cv2.createTrackbar("Threshold3","Parameters",45,255,empty)
cv2.createTrackbar("Threshold4","Parameters",645,1000,empty)
cv2.createTrackbar("Threshold5","Parameters",645,1000,empty)

frame = cv2.imread("C:\\Users\\janra\Desktop\\DIR2024\\black.jpg")
frame = frame[50:380, 250:520]
while True:
    imgContour = frame.copy()
    Threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    Threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    Threshold3 = cv2.getTrackbarPos("Threshold3", "Parameters")
    Threshold4 = cv2.getTrackbarPos("Threshold4", "Parameters")
    Threshold5 = cv2.getTrackbarPos("Threshold5", "Parameters")
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_black = np.array([Threshold1,Threshold2,Threshold3])
    upper_black = np.array([0,0,0])
    mask = cv2.inRange(hsv,upper_black,lower_black)

    image2 = cv2.Canny(mask, 645 ,645)
    kernel = np.ones((5,5))
    image2 = cv2.dilate(image2,kernel,iterations=1)

    getContours(image2,imgContour,Threshold5)

    cv2.imshow("maska", mask)
    cv2.imshow("1",frame)
    cv2.imshow("canny",imgContour)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break