import cv2

faceCascade= cv2.CascadeClassifier("\\face detection\haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (512, 512))
    faces = faceCascade.detectMultiScale(img,1.1,4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
