from position_detector import points
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
from tracker import Tracker
import random

# Yolo Model Initialization
model = YOLO('../yolov8s.pt')

# Detect and use Live Camera for Video Capture
video_path = '../assets/Videos/dropout.mp4'
video = cv2.VideoCapture(video_path)

# Frame Width and Height
frame_width = int(video.get(3))
frame_height = int(video.get(4))

# Custom Class List from COCO File
my_file = open("../assets/files/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# To Get Window points Custom Class
cv2.setMouseCallback('VisionSense', points)

tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]


pos_area = [(627, 375), (574, 583), (978, 643), (1001, 407)]
staff_area = [(569, 602), (530, 752), (957, 813), (971, 656)]

pos_count = 0
staff_count = 0

detection_threshold = 0.5

while video.isOpened():
    # frame refers Current Image Frame from Video
    ret, image = video.read()

    # If the image was not read successfully, end the loop
    if not ret:
        break

    # Resize Image according to the requirements
    image = cv2.resize(image, (1920, 1080))

    # Yolov8 Package Results of each Image Frame
    # yolov8Results = model.predict(image)

    # Yolov8 Package Results of each Image Frame Custom Classes
    yolov8Results = model.predict(source=image, show=False, stream=True, classes=[0])

    for i, (result) in enumerate(yolov8Results):
        a = result.boxes.data
        px = pd.DataFrame(a).astype("float")

        # Show Custom Boxes in Each Object Detection
        detections = []

        # Appending each object position to custom List
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score,class_id])

        tracker.update(image, detections)
        staff_count = pos_count = 0

        for track in tracker.tracks:
            x1, y1, x2, y2 = track.bbox
            track_id = track.track_id
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2

            results = cv2.pointPolygonTest(np.array(pos_area, np.int32), (cx, cy), False)

            if results >= 0:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                cv2.putText(image, str(int(track_id)), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, 2,
                            (colors[track_id % len(colors)]), 2)
                pos_count += 1

            results1 = cv2.pointPolygonTest(np.array(staff_area, np.int32), (cx, cy), False)

            if results1 >= 0:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                cv2.putText(image, str(int(track_id)), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, 2,
                            (colors[track_id % len(colors)]), 2)
                staff_count += 1


    cv2.putText(image, str(f'Staff Count: {int(staff_count)}'), (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(image, str(f'POS Count: {int(pos_count)}'), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.polylines(image, [np.array(pos_area, np.int32)], True, (0, 0, 255), 3)
    cv2.polylines(image, [np.array(staff_area, np.int32)], True, (255, 0, 0), 3)

    # Show Image
    cv2.imshow('VisionSense', image)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
