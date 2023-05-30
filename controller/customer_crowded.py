from controller.save_frame_info_to_excel import save_frame_info_to_excel
from controller.send_whats_app_notification import send_whatsapp_message
from position_detector import points
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
from tracker import Tracker
import random
import threading


# Yolo Model Initialization
model = YOLO('../yolov8s.pt')

# Detect and use Live Camera for Video Capture
video_path = '../assets/Videos/crouded.mp4'
video = cv2.VideoCapture(video_path)

# Frame Width and Height
frame_width = int(video.get(3))
frame_height = int(video.get(4))

# Define the output video path and properties
output_path = '/Users/mukeshnaidu/MukeshGit/output/crouded.mp4'
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

checkout_area1 = [(499, 188), (842, 1051), (1909, 790), (1165, 6)]

# Initialize a dictionary to store the count of track IDs
track_count = {}

detection_threshold = 0.5

averageWaitingTime = 5
notificationSent = 0
pos = 3

# Initialize a frame counter
frame_count = 0

while video.isOpened():
    # frame refers Current Image Frame from Video
    ret, image = video.read()

    # If the image was not read successfully, end the loop
    if not ret:
        break

    # Increment the frame counter
    frame_count += 1

    # Resize Image according to the requirements
    #image = cv2.resize(image, (1920, 1080))

    # Yolov8 Package Results of each Image Frame
    # yolov8Results = model.predict(image)

    # Yolov8 Package Results of each Image Frame Custom Classes
    yolov8Results = model.predict(source=image, show=False, stream=False, classes=[0])

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
                detections.append([x1, y1, x2, y2, score, class_id])

        tracker.update(image, detections)

        # Create a set to store the track IDs detected in the current frame
        current_frame_track_ids = set()

        for track in tracker.tracks:
            x1, y1, x2, y2 = track.bbox
            track_id = track.track_id
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2

            results = cv2.pointPolygonTest(np.array(checkout_area1, np.int32), (cx, cy), False)
            if results >= 0:

                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                cv2.putText(image, str(int(track_id)), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, 2,
                            (colors[track_id % len(colors)]), 2)

                # Add the track ID to the set for the current frame
                current_frame_track_ids.add(track_id)

                # Update the count for each track ID
                if track_id not in track_count:
                    track_count[track_id] = 1
                else:
                    track_count[track_id] = track_count.get(track_id, 0) + 1

        # Reduce the count for track IDs that are not detected in the current frame
        for track_id in list(track_count.keys()):
            if track_id not in current_frame_track_ids:
                track_count[track_id] -= 1
                if track_count[track_id] < 0:
                    del track_count[track_id]

    cv2.polylines(image, [np.array(checkout_area1, np.int32)], True, (0, 0, 255), 3)

    # Display the track ID and count on the image
    cv2.putText(image, f"Live Customer Count: {len(current_frame_track_ids)}", (35, 819), cv2.FONT_HERSHEY_COMPLEX, 1,
                (245, 248, 252), 2)

    if len(current_frame_track_ids) <= 4:
        averageWaitingTime = 3
    elif 5 <= len(current_frame_track_ids) <= 6:
        averageWaitingTime = 4
    elif 7 <= len(current_frame_track_ids) <= 8:
        averageWaitingTime = 6
    if len(current_frame_track_ids) >= 9:
        averageWaitingTime = 8

    # Example usage
    account_sid = 'ACa5e43dc25802687dfe14ac2df99bf5f4'
    auth_token = '0b22b5f49f0c0e33412b0ea48edd7ed7'
    from_number = '+14155238886'
    to_number = '+971529123587'
    image_path = 'https://firebasestorage.googleapis.com/v0/b/landmarkhackathon.appspot.com/o/ssco_croud.jpg?alt=media&token=d98d0c88-bd58-4df0-9dc5-0666b95b72f5'
    text_message = 'It appears that there is a large crowd, and one of the points of sale (POS) is not active. To minimize the average waiting time, kindly organize a cashier to assist.'
    if frame_count >= 15 and len(current_frame_track_ids) >= 5 and notificationSent == 0:
        file_path = f"/Users/mukeshnaidu/MukeshGit/output/ssco_croud.jpg"
        whatsapp_thread = threading.Thread(target= send_whatsapp_message, args=(
        account_sid, auth_token, from_number, to_number, text_message, image_path))
        whatsapp_thread.start()
        notificationSent = 1
        # cv2.imwrite(file_path, image)

    if notificationSent == 1:
        cv2.putText(image, f"Notification Triggered to manager : {1}", (35, 919),
                    cv2.FONT_HERSHEY_COMPLEX, 1,
                    (245, 248, 252), 2)
    else:
        cv2.putText(image, f"Notification Triggered to manager : {0}", (35, 919),
                    cv2.FONT_HERSHEY_COMPLEX, 1,
                    (245, 248, 252), 2)

    cv2.putText(image, f"Average Waiting Time : {averageWaitingTime} minutes", (35, 1019),
                cv2.FONT_HERSHEY_COMPLEX, 1,
                (245, 248, 252), 2)

    # Write the frame to the output video
    output_video.write(image)

    # Show Image
    cv2.imshow('VisionSense', image)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
