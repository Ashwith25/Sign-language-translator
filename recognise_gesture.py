import numpy as np
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
from cvzone.HandTrackingModule import HandDetector
import cvzone
# import tensorflow as tf

word_dict = {0:'One', 1:'Two', 2:'Three'}

model = keras.models.load_model("signModel")
background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

def cal_accum_avg(frame, accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype("float")
        return None

    cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold=25):
    global background
    # diff = cv2.absdiff(background.astype("uint8"), frame)
    _ , thresholded = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
    # contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if len(contours) == 0:
    #     return None
    # else:
        # hand_segment_max_cont = max(contours, key=cv2.contourArea)
    return thresholded

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = HandDetector(detectionCon=0.8, maxHands=2)
num_frames = 0
while True:

    ret, frame = cam.read()
    frame = cv2.flip(frame, 1)
    frame_copy = frame.copy()

    hands, img = detector.findHands(frame)  # with draw
    # hands = detector.findHands(img, draw=False)  # without draw

    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"]  # List of 21 Landmark points
        bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
        centerPoint1 = hand1['center']  # center of the hand cx,cy
        handType1 = hand1["type"]  # Handtype Left or Right

        fingers1 = detector.fingersUp(hand1)

        ROI_top = bbox1[1] - 20
        ROI_bottom = bbox1[3] + bbox1[1] + 20
        ROI_right =  bbox1[0] - 20
        ROI_left = bbox1[2] + bbox1[0] + 20

    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
    # roi = frame[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]

    # gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
    gray_frame = frame

    if num_frames < 70:
        pass
        # cal_accum_avg(gray_frame, accumulated_weight)
        # cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    else:
        hand = segment_hand(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        if hand is not None:
            thresholded = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            (thresh, thresholded) = cv2.threshold(thresholded, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            # cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255,128,0), 2)
            cv2.imshow("Thresholded Hand Image", thresholded)
            
            thresholded = cv2.resize(thresholded, (64, 64))
            thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
            thresholded = np.reshape(thresholded, (1,thresholded.shape[0],thresholded.shape[1],3))

            pred = model.predict(thresholded)
            cv2.putText(frame_copy, word_dict[np.argmax(pred)], (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
            
    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)
    num_frames += 1
    # cv2.imshow("Sign Detection", frame_copy)

    imgStacked = cvzone.stackImages([img, frame_copy], 2, 1)
    cv2.imshow("Image", imgStacked)
    # cv2.imshow("Image", img)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()