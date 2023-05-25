from controller.save_frame_info_to_excel import save_frame_info_to_excel, save_ssco_info_to_excel
from position_detector import points
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
from tracker import Tracker
import random
import time
import datetime
import threading
import json

pd.reset_option('all')

# Yolo Model Initialization
model = YOLO('../yolov8s.pt')

# Detect and use Live Camera for Video Capture
video_path = '../assets/Videos/dropout.mp4'
video = cv2.VideoCapture(video_path)

# Frame Width and Height
frame_width = int(video.get(3))
frame_height = int(video.get(4))

# Define the output video path and properties
output_path = '/Users/mukeshnaidu/MukeshGit/output/selfcheckout_1.mp4'
output_codec = cv2.VideoWriter_fourcc(*'mp4v')
output_fps = video.get(cv2.CAP_PROP_FPS)
output_video = cv2.VideoWriter(output_path, output_codec, output_fps, (frame_width, frame_height))

# Custom Class List from COCO File
my_file = open("../assets/files/coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# To Get Window points Custom Class
cv2.setMouseCallback('VisionSense', points)

tracker = Tracker()
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

track_timings = {}

checkout_area1 = [(627, 375), (574, 583), (978, 643), (1001, 407)]
checkout_area2 = [(569, 602), (530, 752), (957, 813), (971, 656)]
checkout_area3 = [(528, 792), (480, 985), (919, 1040), (948, 847)]

camera_id = 1
camera_name = 'SSCO'

detection_threshold = 0.5

# Set the threshold for considering a customer as "left"
left_threshold = 200


def draw_text(image, text, position, color):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)


def draw_bbox(image, bbox, track_id, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
    cv2.putText(image, str(int(track_id)), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, 2, color, 2)


# def save_frame_info_async(track_id, zone, frame_count):
#     current_time = time.time()
#     entry_time = track_timings[track_id]
#     time_spent = current_time - entry_time
#
#     # Convert entry_time to datetime object
#     entry_datetime = datetime.datetime.fromtimestamp(entry_time)
#     save_ssco_info_to_excel(camera_id, camera_name, frame_count, entry_datetime, datetime.datetime.now(), time_spent, track_id, zone)


def save_frame_info_async(track_id, zone, frame_count):
    current_time = time.time()
    entry_time = track_timings[track_id]
    time_spent = current_time - entry_time

    # Convert entry_time to datetime object
    entry_datetime = datetime.datetime.fromtimestamp(entry_time)

    # Create a dictionary with the information
    info = {
        "camera_id": camera_id,
        "camera_name": camera_name,
        "frame_count": frame_count,
        "entry_datetime": str(entry_datetime),
        "exit_datetime": str(datetime.datetime.now()),
        "time_spent": time_spent,
        "track_id": track_id,
        "zone": zone
    }

    # Convert dictionary to JSON format
    json_data = json.dumps(info)

    # Specify the file path
    file_path = "/Users/mukeshnaidu/MukeshGit/output/ssco_info.txt"

    # Write JSON data to a text file
    with open(file_path, "a") as file:
        file.write(json_data + "\n")

def draw_text(image, text, position, color):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)


def process_detection(image, detections, checkout_area, zone):
    global frame_count
    global drop_out  # Declare drop_out as a global variable
    for track in tracker.tracks:
        x1, y1, x2, y2 = track.bbox
        track_id = track.track_id
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2

        results = cv2.pointPolygonTest(np.array(checkout_area, np.int32), (cx, cy), False)
        frame_count += 1

        if results >= 0:
            draw_bbox(image, (x1, y1, x2, y2), track_id, colors[track_id % len(colors)])

            if track_id in track_timings:
                current_time = time.time()
                entry_time = track_timings[track_id]
                time_spent = current_time - float(entry_time)

                # # Check if entry_time is not 'D' (indicating it has been counted as dropout)
                # if entry_time != 'D':
                #     time_spent = current_time - float(entry_time)
                #
                #     # Check if track_id spent more than 50 seconds in checkout_area2 or checkout_area3
                #     if time_spent > 92 and zone in ['B', 'C'] and track_timings[track_id] != 'A':
                #         if track_timings[track_id] != 'D':
                #             drop_out += 1
                #             track_timings[track_id] = 'D'  # Mark track_id as counted for dropout
                # else:
                # time_spent = 0  # Initialize time_spent with 0

                if zone == 'A':
                    text_position = (1381, 892)
                elif zone == 'B':
                    text_position = (1381, 986)
                else:
                    text_position = (1380, 1046)

                draw_text(image, f"Zone {zone} : User {track_id} spent {int(time_spent)} sec", text_position,
                          (24, 25, 26))

                threading.Thread(target=save_frame_info_async, args=(track_id, zone, frame_count)).start()

            else:
                track_timings[track_id] = time.time()


frame_count = 0
drop_out = 0


def process_frame(image):
    global frame_count

    image = cv2.resize(image, (1920, 1080))

    yolov8Results = model.predict(source=image, show=False, stream=False, classes=[0])

    for i, (result) in enumerate(yolov8Results):
        a = result.boxes.data
        px = pd.DataFrame(a).astype("float")

        detections = []

        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            if score > detection_threshold:
                detections.append([x1, y1, x2, y2, score, class_id])

        tracker.update(image, detections)

        process_detection(image, detections, checkout_area1, 'A')
        process_detection(image, detections, checkout_area2, 'B')
        process_detection(image, detections, checkout_area3, 'C')

    cv2.putText(image, f"Drop Out : {drop_out}", (1619, 374), cv2.FONT_HERSHEY_COMPLEX, 1, (24, 25, 26), 2)
    cv2.polylines(image, [np.array(checkout_area1, np.int32)], True, (3, 173, 9), 3)
    cv2.polylines(image, [np.array(checkout_area2, np.int32)], True, (232, 221, 7), 3)
    cv2.polylines(image, [np.array(checkout_area3, np.int32)], True, (232, 75, 7), 3)

    cv2.imshow('VisionSense', image)
    #output_video.write(image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False

    return True


while video.isOpened():
    ret, image = video.read()

    if not ret:
        break

    if not process_frame(image):
        break

video.release()
output_video.release()
cv2.destroyAllWindows()
