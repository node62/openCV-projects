import cv2
import numpy as np
import math

cap = cv2.VideoCapture(r'C:\Users\Theodore Regimon\PycharmProjects\pythonProject\A Line Following Algorithm\cropLineSample.mp4')
width = int(cap.get(3))
height = int(cap.get(4))
def find_center_of_mass(img, contour):
    moments = cv2.moments(contour)
    center_x = int(moments["m10"] / moments["m00"])
    center_y = int(moments["m01"] / moments["m00"])
    cv2.circle(img, (center_x, center_y), 5, (0, 255, 0), 3)
    return center_x, center_y
def getContours(img, imgOg):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max=0
    mcnt=0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max:
            max = area
            mcnt = cnt
    k, l=find_center_of_mass(imgOg, mcnt)
    cv2.drawContours(imgOg, mcnt, -1, (255, 0, 0), 3)
    return imgOg, k, l
iter=0
x1 = y1 = l1 = 0
while True:
    _, imgOg = cap.read()
    img = imgOg.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    imgDilation = cv2.dilate(img, (3, 3), iterations=1)
    imgErosion = cv2.erode(img, (3, 3), iterations=1)
    img, x2, y2 = getContours(img, imgOg)
    cv2.arrowedLine(imgOg, (x1,y1+50),(x2, y2), (0, 0, 255), 5)
    m = x2-x1
    if x1==x2:
        m=0.00009
    slope=(y2-y1)/(m)
    label=(math.atan(slope)/3.14*180)
    angVel=((label-l1)/(1000))
    if(angVel<0):
        angVel=-angVel
    if iter%50==0:
        l1=label
    cv2.putText(imgOg, str(angVel), (5, height-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("Video", img)
    if cv2.waitKey(1)&0xFF == ord(' '):
        break
    if iter%2==0:
        x1=x2
        y1=y2


    iter=iter+1

cv2.destroyAllWindows()
