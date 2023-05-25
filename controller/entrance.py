import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker
import random
from position_detector import points

# Yolo Model Initialization
model = YOLO('../yolov8s.pt')

# Detect and use Live Camera for Video Capture
video_path = '../assets/Videos/entrance.mp4'
video = cv2.VideoCapture(video_path)

# Frame Width and Height
frame_width = int(video.get(3))
frame_height = int(video.get(4))

# Define the output video path and properties
output_path = '/Users/mukeshnaidu/MukeshGit/output/entrance.mp4'
output_codec = cv2.VideoWriter_fourcc(*'mp4v')
output_fps = video.get(cv2.CAP_PROP_FPS)
output_video = cv2.VideoWriter(output_path, output_codec, output_fps, (frame_width, frame_height))

# Custom Class List from COCO File
my_file = open("../assets/files/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


customerIn = 0
customerOut = 3

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

area1 = [(1589, 402), (1682, 1049), (1699, 1052), (1607, 389)]
area2 = [(872, 167), (912, 487), (919, 488), (877, 168)]

customerIn = 0
customerOut = 3

count = 0

frame_count = 0

while video.isOpened():
    # frame refers Current Image Frame from Video
    ret, image = video.read()

    # If the image was not read successfully, end the loop
    if not ret:
        break

    # Increment the frame counter
    frame_count += 1

    # Yolov8 Package Results of each Image Frame Custom Classes
    yolov8Results = model.predict(source=image, show=False, stream=False, classes=[0])

    cv2.polylines(image, [np.array(area1, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(image, f'Customer IN: {int(customerIn)}', (845, 35), cv2.FONT_HERSHEY_COMPLEX, 0.5, (245, 248, 252), 1)

    cv2.polylines(image, [np.array(area2, np.int32)], True, (255, 0, 255), 2)
    cv2.putText(image, f'Customer OUT: {int(customerOut)}', (845, 65), cv2.FONT_HERSHEY_COMPLEX, 0.5, (245, 248, 252),
                1)

    # Write the frame to the output video
    output_video.write(image)

    cv2.imshow("RGB", image)
    if cv2.waitKey(1) == 27:  # Press Esc key to exit
        break
