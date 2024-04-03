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
import lcm


class Camera:
    def __init__(self, camera_id, width, height):
        self.cam_id = camera_id
        self.cap = Picamera2(self.cam_id)
        self.skip_frames = 5  # Process every 5th frame for tag detection
        self.w = width
        self.h = height
        self.cap.preview_configuration.main.size = (self.w, self.h)
        self.cap.preview_configuration.main.format = "RGB888"
        self.cap.preview_configuration.align()
        self.cap.configure("preview")
        self.cap.start()

        self.frame_count = 0
        self.detections = dict()
        self.load_calibration_data()
        self.lcm = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
    
    def load_calibration_data(self):
        with open(camera_intrinsics_filepath, 'r') as f:
            params = yaml.safe_load(f)

        self.camera_matrix = np.array(params['camera_matrix'], dtype=np.float32)
        self.distortion_coefficients = np.array(params['distortion_coefficients'], dtype=np.float32)
        self.new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.distortion_coefficients, (self.w, self.h), 1, (self.w, self.h))
        self.x_undistored, self.y_undistorted, self.w, self.h = roi

    def detect(self):
        while True:
            frame = picam.capture_array()
            if not frame == None:
                break
            self.frame_count += 1
            # Process for tag detection only every 5th frame
            if self.frame_count % self.skip_frames == 0:

                self.detections = self.detector.detect(frame) # TODO: fix
            
                self.publish_cones()

    # def publish_cones(self):
    #     """
    #     Publish the apriltag message
    #     """
    #     # msg = mbot_apriltag_array_t()
    #     # msg.array_size = len(self.detections)
    #     # msg.detections = []
    #     if msg.array_size > 0:
    #         for detect in self.detections:
    #             # Pose estimation for detected tag
    #             image_points = np.array(detect['lb-rb-rt-lt'], dtype=np.float32)
    #             if detect['id'] < 10: # big tag
    #                 retval, rvec, tvec = cv2.solvePnP(self.object_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)

    #             if detect['id'] > 10: # small tag at center
    #                 retval, rvec, tvec = cv2.solvePnP(self.small_object_points, image_points, self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)

    #             # Convert rotation vector  to a rotation matrix
    #             rotation_matrix, _ = cv2.Rodrigues(rvec)
         
    #             # # Calculate Euler angles: roll, pitch, yaw - x, y, z in degrees
    #             # for apriltag, x is horizontal, y is vertical, z is outward
    #             roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation_matrix)
    #             quaternion = rotation_matrix_to_quaternion(rotation_matrix)

    #             apriltag = mbot_apriltag_t()
    #             apriltag.tag_id = detect['id']
    #             apriltag.pose.x = tvec[0][0]
    #             apriltag.pose.y = tvec[1][0]
    #             apriltag.pose.z = tvec[2][0]
    #             apriltag.pose.angles_rpy = [roll, pitch, yaw]
    #             apriltag.pose.angles_quat = quaternion
    #             msg.detections.append(apriltag)

    #     self.lcm.publish("MBOT_APRILTAG_ARRAY", msg.encode())

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
    print("No camera intrinsics found, cannot extract depth. Terminating...")
    exit()

# configure rpi camera

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

center_pixel_line = w / 2

while True:
    try:
        im = picam.capture_array()

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

            height = np.abs(y2 - y1)
            cone_height_sensor = height / m_scaled
            distance_cm = int((cone_height_real * arducam_focal_length_mm / cone_height_sensor) / 10)
            distance_str = str(str(distance_cm) + "cm")

            dist_from_center = (x1 + x2) / 2
            heading_difference = round(np.arctan((dist_from_center - center_pixel_line) / arducam_focal_length_pixels), 4)

            range_heading_info = "R: " + distance_str + ', H: ' + str(heading_difference)

            cvzone.putTextRect(im, range_heading_info, (x2,y2), 1,1)

            cv2.rectangle(im, (x1,y1), (x2,y2),(0,0,255),2)
            cvzone.putTextRect(im,f'{c}',(x1,y1),1,1)

        cv2.imshow("camera",im)
        if cv2.waitKey(1)==ord('q'):
            break
    except KeyboardInterrupt:
        break

print("Terminating...")
cv2.destroyAllWindows()
picam.stop()
