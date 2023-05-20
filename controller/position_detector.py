import cv2


# To Get Window points
def points(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colors_bgr = [x, y]
        print(colors_bgr)


cv2.namedWindow('VisionSense')
