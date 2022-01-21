import keras
from cvzone.HandTrackingModule import HandDetector
import cvzone

import av
import cv2

import numpy as np
import streamlit as st

from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def main():
    st.header("Sign Language Translation using Human-Computer Interaction")
    st.markdown(f'<h1 style="color:#33ff33; font-size:24px;">{"Indian Sign Language Translator"}</h1>', unsafe_allow_html=True)

    # sign_language_det = "Sign Language Live Detector"
    # app_mode = st.sidebar.selectbox( "Choose the app mode",
    #     [
    #         sign_language_det
    #     ],
    # )

    sign_language_detector()

def sign_language_detector():

    class OpenCVVideoProcessor(VideoProcessorBase):

        def __init__(self):
            self.detector = HandDetector(detectionCon=0.8, maxHands=2)
            self.ROI_top = 100
            self.ROI_bottom = 300
            self.ROI_right = 150
            self.ROI_left = 350
            self.background = None
            self.accumulated_weight = 0.5   

        def cal_accum_avg(self, frame, accumulated_weight):
            global background
            if self.background is None:
                background = frame.copy().astype("float")
                return None

            cv2.accumulateWeighted(self, frame, self.background, accumulated_weight)

        def segment_hand(self, frame, threshold=25):
            global background
            _ , thresholded = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                return None
            return thresholded


        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            word_dict = {0:'One', 1:'Two', 2:'Three', 3:'I Love You', 4:'Little'}
            model = keras.models.load_model("signModelNew")

            while True:
                img = cv2.flip(img,1)
                hands, img = self.detector.findHands(img)  # with draw

                if hands:
                    hand1 = hands[0]
                    bbox1 = hand1["bbox"]

                    self.ROI_top = bbox1[1] - 20
                    self.ROI_bottom = bbox1[3] + bbox1[1] + 20
                    self.ROI_right =  bbox1[0] - 20
                    self.ROI_left = bbox1[2] + bbox1[0] + 20

                roi = img[self.ROI_top:self.ROI_bottom, self.ROI_right:self.ROI_left]

                hand = self.segment_hand(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                if hand is not None:
                    thresholded = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    (_, thresholded) = cv2.threshold(thresholded, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    
                    thresholded = cv2.resize(thresholded, (64, 64))
                    thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
                    thresholded = np.reshape(thresholded, (1,thresholded.shape[0],thresholded.shape[1],3))

                    pred = model.predict(thresholded)
                    predText = word_dict[np.argmax(pred)]
                    cv2.putText(img, predText, (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
                        
                cv2.rectangle(img, (self.ROI_left, self.ROI_top), (self.ROI_right, self.ROI_bottom), (255,128,0), 3)
                
                return av.VideoFrame.from_ndarray(img,format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=OpenCVVideoProcessor,
        async_processing=True,
    )

if __name__ == "__main__":
    main()