from PIL import ImageGrab
from threading import Thread
import sys
import numpy as np
import cv2
import pyscreenshot

import os

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800,600)

# Load Yolo
net = cv2.dnn.readNet("C:\\Users\\staag\\OneDrive\\Documents\\r6\\yolov3.weights", "C:\\Users\\staag\\OneDrive\\Documents\\r6\\yolov3.cfg")
classes = []
with open("C:\\Users\\staag\\OneDrive\\Documents\\r6\\coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

class VideoStream:
    def __init__(self):
        self.frame = pyscreenshot.grab(bbox = (5,60,400,300 + 32))
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self
 
    def update(self):
        while True:
            if self.stopped:
                return
            self.frame = pyscreenshot.grab(bbox = (5,60,400,300 + 32))
 
    def read(self):
        return self.frame
 
    def stop(self):
        self.stopped = True

stream = VideoStream()

while True:
    img = stream.read()
    img_np = np.array(img)
    height, width, channels = img_np.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img_np, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                print('detected')
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                       
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(img_np, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img_np, label, (x, y + 30), font, 3, color, 3)

    cv2.imshow("image", img_np)
    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
