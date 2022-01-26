import numpy as np
np.set_printoptions(suppress=True)
import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator
from HandTrackingModule import HandDetector
import cvzone
# import tensorflow as tf

import pyttsx3

#audio of system to respond
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)
engine.setProperty('rate',180)

# word_dict = {0:'One', 1:'Two', 2:'Three', 3:'A', 4:'I Love You', 5:'Little'}
word_dict = {0:'One', 1:'Two', 2:'Three', 3: 'A', 4: 'B', 5: 'C', 6: 'D', 7: 'E', 8: 'Hi', 9: 'I Love You'}

model = keras.models.load_model("signForColor")
background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

# def cal_accum_avg(frame, accumulated_weight):
#     global background
#     if background is None:
#         background = frame.copy().astype("float")
#         return None

#     cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold=25):

    global background
    # diff = cv2.absdiff(background.astype("uint8"), frame)
    _ , thresholded = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
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
    # hands = detector.findHands(frame, draw=False)  # without draw

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

        if len(hands) == 2:
            # Hand 2
            hand2 = hands[1]
            # lmList2 = hand2["lmList"]
            bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
            # centerPoint2 = hand2['center']  # center of the hand cx,cy
            handType2 = hand2["type"]

            fingers2 = detector.fingersUp(hand2)

            # Find Distance between two Landmarks. Could be same hand or different hands
            # length, info, img = detector.findDistance(lmList1[8], lmList2[8], img)

            absDiff = abs((bbox1[1] + bbox1[3]) - (bbox2[1] + bbox2[3]))

            ROI_top = bbox1[1] - 20 if bbox1[1] < bbox2[1] else bbox2[1] - 20
            ROI_bottom = bbox1[3] + bbox1[1] + 20 + absDiff if bbox1[1] < bbox2[1] else bbox2[3] + bbox2[1] + 20 + absDiff
            ROI_right =  bbox1[0] - 20 if handType2 == 'Left' else bbox2[0] - 20
            ROI_left = bbox1[2] + (bbox1[0] if handType2 == 'Left' else bbox2[0]) + bbox2[2] + 80


    roi = frame[ROI_top if ROI_top >= 0 else 0 :ROI_bottom, ROI_right if ROI_right >= 0 else 0:ROI_left]
    # print(ROI_top if ROI_top >= 0 else 0, ROI_bottom, ROI_right, ROI_left)
    # roi = frame[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]

    # gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)
    # gray_frame = frame

    if not hands:
        cv2.putText(frame_copy, "No hands detected", (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
        # cal_accum_avg(gray_frame, accumulated_weight)
        # cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    else:
        # print(roi)
        # roi = segment_hand(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        # if hand is not None:
        thresholded = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        thresholded = cv2.GaussianBlur(thresholded, (3, 3), cv2.BORDER_DEFAULT)
        wide = cv2.Canny(thresholded, 60, 240)
        cv2.imshow("med", wide)
        # thresholded = cv2.adaptiveThreshold(thresholded, 255,cv2.ADAPTIVE_THRESH_MEAN_C,\
        #     cv2.THRESH_BINARY,11,2)
        (thresh, thresholded) = cv2.threshold(thresholded, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255,128,0), 2)
        # thresholded = roi
        cv2.imshow("Detected Hand Image", thresholded)
        
        thresholded = cv2.resize(thresholded, (64, 64))
        thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
        thresholded = np.reshape(thresholded, (1, thresholded.shape[0], thresholded.shape[1], 3)) # n, h, w, channels

        pred = model.predict(thresholded)
        # print(pred*100)
        predText = word_dict[np.argmax(pred)]
        cv2.putText(frame_copy, predText, (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
        # speak(predText)
            
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