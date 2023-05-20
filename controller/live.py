from position_detector import points
import cv2
from ultralytics import YOLO
import pandas as pd
from tracker import *

# Yolo Model Initialization
model = YOLO('../yolov8x.pt')

# Detect and use Live Camera for Video Capture
video_path = '../assets/Videos/storelmg.mp4'
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

while video.isOpened():
    # frame refers Current Image Frame from Video
    ret, image = video.read()

    # If the image was not read successfully, end the loop
    if not ret:
        break

    # Resize Image according to the requirements
    image = cv2.resize(image, (1000, 1252))

    # Yolov8 Package Results of each Image Frame
    # yolov8Results = model.predict(image)

    # Yolov8 Package Results of each Image Frame Custom Classes
    yolov8Results = model.predict(source=image, show=False, stream=True, classes=[0, 63, 64, 66])

    for i, (result) in enumerate(yolov8Results):
        a = result.boxes.boxes
        px = pd.DataFrame(a).astype("float")

        # Show Custom Boxes in Each Object Detection
        objectPositionList = []

        # Appending each object position to custom List
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            objectPositionList.append([x1, y1, x2, y2])

        boxes_ids = tracker.update(objectPositionList)

        for bbox in boxes_ids:
            x3, y3, x4, y4, boxId = bbox
            cx = int(x3 + x4) // 2
            cy = int(y3 + y4) // 2
            cv2.rectangle(image, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv2.circle(image, (x4, y4), 5, (255, 0, 255), -1)
            cv2.putText(image, str(int(boxId)), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 1)

    # Show Image
    cv2.imshow('VisionSense', image)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
