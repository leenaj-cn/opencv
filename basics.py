from ctypes import resize
import cv2
import numpy as np

kernel = np.ones((5,5), np.uint8)
kernel1 = np.ones((5,5), np.uint8)

def empty(v):
    pass


def BasicImageProcessing():
    img = cv2.imread("data/pic/cat.jpg")

    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (15,15),10)
    canny = cv2.Canny(img,300,350)
    dilate = cv2.dilate(canny, kernel, iterations=1)
    erode = cv2.erode(dilate, kernel1,iterations=1)    

    cv2.imshow("gray", gray)
    cv2.imshow("canny", canny)
    cv2.imshow("dilate", dilate)
    cv2.imshow("erode", erode)

    cv2.waitKey()

def VideoCapture():
  capture = cv2.VideoCapture("data/video/disco.mp4")
  while True:
    ret, frame = capture.read()
    if ret:  
        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        hsvimg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        cv2.imshow("hsv,",hsvimg)
        cv2.imshow("frame,",frame)
        
        cv2.waitKey(10)
    else:
        break

def Draw():
    #draw image
    drawimg = np.zeros((600,600,3), np.uint8)
    cv2.line(drawimg, (0,0), (drawimg.shape[1],drawimg.shape[0]), (255,0,255),1)
    cv2.rectangle(drawimg, (0,0), (300,200), (255,0,0), 2)
    cv2.rectangle(drawimg, (0,0), (300,200), (0,255,0), cv2.FILLED)
    cv2.circle(drawimg, (400,200), 100, (255,255,0), cv2.FILLED)
    #cv2.putText(drawimg, "test")

    cv2.imshow("draw", drawimg)
    cv2.waitKey()

def ExtractColor():
    #extract color
    img = cv2.imread("data/pic/cat.jpg")
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    
    #create trackbar
    cv2.namedWindow('TrackBar')
    cv2.resizeWindow('TrackBar', 640, 320)

    cv2.createTrackbar("Hue Min", "TrackBar", 0, 179, empty)
    cv2.createTrackbar("Hue Max", "TrackBar", 179, 179, empty)
    cv2.createTrackbar("Sat Min", "TrackBar", 0, 255, empty)
    cv2.createTrackbar("Sat Max", "TrackBar", 255, 255, empty)
    cv2.createTrackbar("Val Min", "TrackBar", 0, 255, empty)
    cv2.createTrackbar("Val Max", "TrackBar", 255, 255, empty)

    #hsv image
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    while True:
        h_min = cv2.getTrackbarPos('Hue Min', 'TrackBar')
        h_max = cv2.getTrackbarPos('Hue Max', 'TrackBar')
        s_min = cv2.getTrackbarPos('Sat Min', 'TrackBar')
        s_max = cv2.getTrackbarPos('Sat Max', 'TrackBar')
        v_min = cv2.getTrackbarPos('Val Min', 'TrackBar')
        v_max = cv2.getTrackbarPos('Val Max', 'TrackBar')
        print(h_min, h_max)

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        mask = cv2.inRange(hsvimg, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow("hsv", hsvimg)
        cv2.imshow("mask", mask)
        cv2.imshow("result", result)
        cv2.imshow("img", img)

        cv2.waitKey(1)

def EdgeDetect():
    img = cv2.imread("data/pic/girl.png")
    imgContour = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(img,150,200)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 4)
        area = cv2.contourArea(cnt)
        print (f"area: {area}")
        if area > 3:
            length = cv2.arcLength(cnt, True)
            print (f"length: {length} ")
            vertices = cv2.approxPolyDP(cnt, length*0.02, True)
            print(f"vertices: {len(vertices)}")

            x,y,w,h = cv2.boundingRect(vertices)
            cv2.rectangle(imgContour, (x,y), (x+w, y+h), (0,255,0), 4)
            if len(vertices) > 7:
                cv2.putText(imgContour, 'curve line', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)


    cv2.imshow("canny", canny)
    cv2.imshow("imgContour", imgContour)
    cv2.waitKey()

def FaceDetect():
    img = cv2.imread("data/pic/lenna.png")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCasecade = cv2.CascadeClassifier('data/model/haarcascade_frontalface_alt.xml')
    faceRect = faceCasecade.detectMultiScale(gray, 1.1, 5)
    print(f"detect {len(faceRect)} faces")

    for (x, y, w, h) in faceRect:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)

    cv2.imshow("img", img)
    cv2.waitKey()


if __name__ == "__main__":
    BasicImageProcessing()
    VideoCapture()
    Draw()
    ExtractColor()
    EdgeDetect()
    FaceDetect()






 



