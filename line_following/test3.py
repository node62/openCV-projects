# using all the mid points between two y coordinates, then creating regression line of those mid points

import cv2
import numpy as np
import math

cap = cv2.VideoCapture(r'C:\Users\Theodore Regimon\PycharmProjects\pythonProject\A Line Following Algorithm\cropLineSample.mp4')
kernel = np.ones((5,5),np.uint8)
tp=10
width = int(cap.get(3))
height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('output_video.mp4', fourcc, 30, (width, height))

def findMidPoint(img, y2):
    mat = np.array(img)
    x1=0
    x2=width-1

    for j in range(0, width, 1):
        if mat[y2][j]==255:
            x1=j
            break

    for j in range(width-1, -1, -1):
        if mat[y2][j]==255:
            x2=j
            break

    def check(a, b):
        diff=5
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
    return mid

def getContours(img, imgOg):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max=0
    mcnt=0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max:
            max = area
            mcnt = cnt
    cv2.drawContours(imgOg, mcnt, -1, (255, 0, 0), 1)
    return imgOg

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

    h1 = round(0.6*height)
    h2 = round(0.9*height)

    n = h2-h1+1
    arr = [0 for _1 in range(n)]

    for i in range(n):
        centre = round(findMidPoint(img, i+h1))
        arr[i] = [centre, i+h1]
        cv2.circle(imgOg, arr[i], 1, (0,255,0), 1)

    sumX = 0
    sumY = 0
    for [x, y] in arr:
        sumX += x
        sumY += y

    meanX = sumX/n
    meanY = sumY/n

    num = 0
    dem = 0
    for [x, y] in arr:
        num = (meanX-x)*(meanY-y)
        den = (meanY-y)*(meanY-y)

    slope = num/den
    xInter = meanX - slope * meanY

    starting_point = (round(xInter), 0)
    ending_point = (round(slope*height+xInter), height)

    cv2.line(imgOg, starting_point, ending_point, (0,0,255), thickness=2)


    cv2.imshow("Video", imgOg)
    cv2.imshow("canny", contimg)
    output_video.write(imgOg)
    if cv2.waitKey(1)&0xFF == ord(' '):
        break
    iter=iter+1
    # print("++++++++++++++++++++++++++++++++++++++++++")

cap.release()
output_video.release()
cv2.destroyAllWindows()
