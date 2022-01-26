import cv2
import numpy as np
from HandTrackingModule import HandDetector
import cvzone

background = None
accumulated_weight = 0.5

ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350

# def cal_accum_avg(frame, accumulated_weight):
#     global background
#     if background is None:
#         background = frame.copy().astype("float")
#         return None
        
#     cv2.accumulateWeighted(frame, background, accumulated_weight)

def segment_hand(frame, threshold=25):
    # global background
    # diff = cv2.absdiff(background.astype("uint8"), frame)
    # _ , thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    # image, contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours) == 0:
    #     return None
    # else:
    #     hand_segment_max_cont = max(contours, key=cv2.contourArea)
    #     return (thresholded, hand_segment_max_cont)

    global background
    # diff = cv2.absdiff(background.astype("uint8"), frame)
    _ , thresholded = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    # else:
        # hand_segment_max_cont = max(contours, key=cv2.contourArea)
    return thresholded

cam = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

num_frames = 0
element = 'Hi'
num_imgs_taken = 0

while True:
    # ret, frame = cam.read()
    # frame = cv2.flip(frame, 1)
    # frame_copy = frame.copy()
    # roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

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

    # roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]
    roi = frame[ROI_top if ROI_top >= 0 else 0 :ROI_bottom, ROI_right if ROI_right >= 0 else 0:ROI_left]

    # gray_frame = frame

    # gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    if num_frames < 60:
        pass
        # cal_accum_avg(gray_frame, accumulated_weight)
        # if num_frames <= 59:
        #     cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            #cv2.imshow("Sign Detection",frame_copy)
         
    elif num_frames <= 100:
        roi = cv2.GaussianBlur(roi, (9, 9), 0)
        hand = segment_hand(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        if hands:
            # thresholded = roi
            (thresh, thresholded) = cv2.threshold(thresholded, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            cv2.putText(frame_copy, str(num_frames)+" For " + str(element), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
            # cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255,128,0), 2)
            cv2.imshow("Detected Hand Image", thresholded)

        # hand = segment_hand(gray_frame)
        # cv2.putText(frame_copy, "Adjust hand...Gesture for " + str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
        # if hand is not None:
        #     thresholded, hand_segment = hand
        #     cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255,128,0),1)
        #     cv2.putText(frame_copy, str(num_frames)+" For " + str(element), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
        #     cv2.imshow("Thresholded Hand Image", thresholded)
    else: 
        # hand = segment_hand(gray_frame)
        hand = segment_hand(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        if hand is not None:

            thresholded = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            (thresh, thresholded) = cv2.threshold(thresholded, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            cv2.putText(frame_copy, str(num_frames), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
            cv2.putText(frame_copy, str(num_imgs_taken) + 'images' +" For " + str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
            # cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255,128,0), 2)
            cv2.imshow("Thresholded Hand Image", thresholded)

            # thresholded, hand_segment = hand
            # cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255,128,0),1)
            # cv2.putText(frame_copy, str(num_frames), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
            # cv2.putText(frame_copy, str(num_imgs_taken) + 'images' +" For " + str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
            
            # cv2.imshow("Thresholded Hand Image", thresholded)

            if num_imgs_taken <= 150:
                cv2.imwrite(r"D:\\gestures\\test\\"+str(element)+"\\" + str(num_imgs_taken) + '.jpg', thresholded)
            else:
                break
            num_imgs_taken +=1
        else:
            cv2.putText(frame_copy, 'No hand detected', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255,128,0), 3)
    cv2.putText(frame_copy, "Hand sign recognition", (10, 20), cv2.FONT_ITALIC, 0.5, (51,255,51), 1)
    
    num_frames += 1

    cv2.imshow("Sign Detection", frame_copy)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cam.release()