from PIL import ImageGrab
import numpy as np
import cv2
import pyscreenshot

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800,600)

import os

# Contains the frontalface haarcascade.
face_cascade = cv2.CascadeClassifier('C:\\Users\\staag\\Downloads\\haarcascade_upperbody.xml')

color = (100,100,255) # BGR
stroke = 2

while True:
    img = pyscreenshot.grab(bbox = (5,60,1024,768 + 32))
    img_np = np.array(img)

    grayscale = cv2.cvtColor(img_np,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale,1.1, 10)

    for (a,b,d,e) in faces:
        print("found")
        roi_gray = grayscale[b:b+e,a:a+d]
        end_cord_a = a + d
        end_cord_b = b + e
        cv2.rectangle(img_np, (a,b), (end_cord_a,end_cord_b),color,stroke)

    cv2.imshow("image", img_np)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
