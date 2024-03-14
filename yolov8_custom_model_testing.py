# YoloV8 Custom Model Testing
import cv2
from picamera2 import Picamera2, Preview
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import time
import libcamera
from pathlib import Path
import io
import yaml

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

camera_intrinsics_filepath = "camera_intrinsics.yaml"
camera_intrinsics_exists = Path(camera_intrinsics_filepath).is_file()

if (camera_intrinsics_exists):

    with open(camera_intrinsics_filepath, 'r') as f:
        params = yaml.safe_load(f)

    camera_matrix = np.array(params['camera_matrix'], dtype=np.float32)
    distortion_coefficients = np.array(params['distortion_coefficients'], dtype=np.float32)

    h = 480
    w = 640
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))
    x, y, w, h = roi

    arducam_focal_length_mm = 3.04 # mm, from arducam spec sheet
    arducam_focal_length_pixels = new_camera_matrix[1,1]
    m = arducam_focal_length_pixels / arducam_focal_length_mm #pixels per mm conversion factor
    m_scaled = h / (720 / m) # m is found from calibration done at 1280x720, so need to scale it to 640x480

    cone_height_real = 80 #mm
else:
    print("No camera intrinsics found, cannot extract depth")

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
# You can change this to "best.pt" which you downloaded from the notebook!
model=YOLO('example_model.pt')

# load model classes
# if your custom model has different classes, you can create a new text file
# This new text file MUST be in the same order as the classes in "data.yaml" which you downloaded from RoboFlow!
my_file = open("example_classes.txt", "r")
data = my_file.read()
class_list = data.split("\n")
count = 0

# the loop will terminate after this much time so it isn't accidentally left running
timeout = 120   # [seconds]

timeout_start = time.time()

while time.time() < timeout_start + timeout:
    if (rpi):
        im = picam.capture_array()

        if (camera_intrinsics_exists):
            im = cv2.undistort(im, camera_matrix, distortion_coefficients, None, new_camera_matrix)
            im = im[y : y + h, x : x + w]
    count += 1
    if (count%3)!=0:
        continue

    # run prediction on the image with yolov8 model
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

        if (camera_intrinsics_exists):
            height = np.abs(y2 - y1)
            cone_height_sensor = height / m_scaled
            distance_cm = int((cone_height_real * arducam_focal_length_mm / cone_height_sensor) / 10)
            distance_str = str(str(distance_cm) + "cm")
            cvzone.putTextRect(im, distance_str, (x2,y2), 1,1)

        cv2.rectangle(im, (x1,y1), (x2,y2),(0,0,255),2)
        cvzone.putTextRect(im,f'{c}',(x1,y1),1,1)

    cv2.imshow("camera",im)
    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()
if (rpi):
    picam.stop()
