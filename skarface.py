import numpy as np
import cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('parojosG.xml')

nose_cascade = cv2.CascadeClassifier('Nariz.xml')
mouth_cascade = cv2.CascadeClassifier('Mouth.xml')

#img = cv2.imread("pics/dom.jpg")
video_capture = cv2.VideoCapture(0)
while True:
    ret, img = video_capture.read()
    b, g, r = cv2.split(img)

    a = np.ones(b.shape, dtype=b.dtype) * 50

    img = cv2.merge((b, g, r, a))

    sunglasses = cv2.imread("pics/sunglasses.png", flags=cv2.IMREAD_UNCHANGED)

    (sunB, sunG, sunR, sunA) = cv2.split(sunglasses)
    sunB = cv2.bitwise_and(sunB, sunB, mask=sunA)
    sunG = cv2.bitwise_and(sunG, sunG, mask=sunA)
    sunR = cv2.bitwise_and(sunR, sunR, mask=sunA)
    sunglasses = cv2.merge([sunB, sunG, sunR, sunA])


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        eyesy = 0
        eyesh = 0
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            eyesy = ey
            eyesh = eh
            eh *= 2
            eyes_zone = img[y+ey:y+ey+eh, x:x+w]
            resized = cv2.resize(sunglasses, (w, eh))
            sb,sg,sr,sa = cv2.split(resized)
            eyes_sung = cv2.add(resized, eyes_zone, mask=255-sa)
            img[y+ey:y+ey+eh, x:x+w] = eyes_sung

        mouth = mouth_cascade.detectMultiScale(roi_gray)
        mouthy = 0
        for(mx,my,mw,mh) in mouth:
            if my+y > y+(h/2):
                cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(0,0,255),2)
                mouthy = my
                break


        nose = nose_cascade.detectMultiScale(roi_gray)
        for(nx,ny,nw,nh) in nose:
            if ny+nh < mouthy and ny > eyesy+eyesh:
                cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,255,255),2)
                break

    cv2.imshow('Skarface', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
