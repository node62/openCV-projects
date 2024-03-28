# using 2 mid points from the edges

import cv2
import numpy as np
import math

cap = cv2.VideoCapture(r'C:\Users\Theodore Regimon\PycharmProjects\pythonProject\A Line Following Algorithm\cropLineSample.mp4')
kernel = np.ones((5,5),np.uint8)
tp=10
l1=0
width = int(cap.get(3))
height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_video.mp4', fourcc, 30, (width, height))

def findMidPoint(img, y2, m1, m2):
    print(f"initial edge points: {m1}, {m2}")
    mat = np.array(img)
    x1=0
    x2=width-1
    ten=15
    if m1-ten<0:
        m1=ten
    if m2+ten>width-1:
        m2=width-1-ten

    for j in range(m1-ten, m2+ten, 1):
        if mat[y2][j]==255:
            x1=m1=j
            break

    for j in range(m2+ten, m1-ten, -1):
        if mat[y2][j]==255:
            x2=m2=j
            break

    def check(a, b):
        diff=10
        for i in range(a-diff, a+1+diff, 1):
            for j in range(b-diff, b+1+diff, 1):
                if i==j:
                    return 1
        else:
            return 0

    if check(x1,x2):
        if x1<mat.shape[1]/2:
            x1=0
        elif x1>mat.shape[1]/2:
            x2=mat.shape[1]-1
    mid = (x1+x2)//2

    m1=x1
    m2=x2

    print(f"final edge points: {m1}, {m2}")
    return mid, m1, m2

def getContours(img, imgOg):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max=0
    mcnt=0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max:
            max = area
            mcnt = cnt
    cv2.drawContours(imgOg, mcnt, -1, (0, 0,255), 1)
    return imgOg

a1=0
a2=width-1
b1=0
b2=width-1
iter=0
while True:
    _, imgOg = cap.read()
    blank=np.zeros_like(imgOg)
    img = imgOg.copy()
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.dilate(img, (3,3), iterations=1)
    img = cv2.erode(img, (3,3), iterations=1)
    # img = imgCanny = cv2.Canny(img, 0, 160)
    img = getContours(img, img)
    contimg = getContours(img, blank)
    p1,a1,a2  = findMidPoint(img, round(0.6*height), a1, a2)
    print(f"P1: {p1}", )
    p2, b1, b2 = findMidPoint(img, round(0.9*height), b1, b2)
    print(f"P2: {p2}", end="\n\n")
    cv2.arrowedLine(imgOg, (p2, 324),(p1, 216), (255, 0, 0), 5)
    # hor = np.hstack((imgOg, imgCanny))
    if p1==p2:
        p1=0
    slope=(324-216)/(p2-p1)
    label=(math.atan(slope)/3.14*180)
    angVel=((label-l1)/(tp*1000))
    if(angVel<0):
        angVel=-angVel
    if iter%100==0:
        l1=label
    cv2.putText(imgOg, str(angVel), (5, height-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    # direction="Right"
    # if p1 < p2:
    #     direction="Left"
    # cv2.putText(imgOg, direction, (5, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Video", imgOg)
    cv2.imshow("canny", contimg)
    output_video.write(imgOg)
    if cv2.waitKey(tp)&0xFF == ord(' '):
        break
    iter=iter+1
cap.release()
output_video.release()
cv2.destroyAllWindows()
