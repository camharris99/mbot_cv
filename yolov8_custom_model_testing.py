# YoloV8 Custom Model Testing
import cv2
from picamera2 import Picamera2, Preview
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import time
import libcamera
import os
import io

# This detects if the computing platform is a raspberry pi
def is_raspberrypi():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            if 'raspberry pi' in m.read().lower(): return True
    except Exception: pass
    return False

# TODO: integrate with jetson
# flag for rpi vs jetson
rpi = is_raspberrypi()

# configure rpi camera
if (rpi):
    cam_id = 0
    picam = Picamera2(cam_id)
    picam.preview_configuration.main.size = (640, 480)
    picam.preview_configuration.main.format = "RGB888"
    picam.preview_configuration.align()
    picam.configure("preview")
    picam.start(show_preview=False)

time.sleep(2)

# load yolo model
model=YOLO('example_model.pt') # You can change this to "best.pt" which you downloaded from the notebook!

# load model classes
my_file = open("example_classes.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count = 0

# the loop will terminate after this much time so it isn't accidentally left running
timeout = 30   # [seconds]

timeout_start = time.time()

while time.time() < timeout_start + timeout:
    if (rpi):
        im = picam.capture_array()
    count += 1
    if (count%3)!=0:
        continue
    results=model.predict(im)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")


    for index,row in px.iterrows():

        # you can extract bounding box size and location using these values
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]

        cv2.rectangle(im, (x1,y1), (x2,y2),(0,0,255),2)
        cvzone.putTextRect(im,f'{c}',(x1,y1),1,1)
    cv2.imshow("camera",im)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()
if (rpi):
    picam.stop()
