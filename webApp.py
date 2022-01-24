import keras
from matplotlib.style import use
from HandTrackingModule import HandDetector
import cvzone

import av
import cv2

import numpy as np
import streamlit as st

import speech_recognition as sr

from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)
import pyaudio
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def takeCommand():
    r = sr.Recognizer()

    audio = pyaudio.PyAudio()
    st.write(audio.get_default_input_device_info())

    with sr.Microphone() as source:
    
        r.adjust_for_ambient_noise(source, duration=2)
        with st.spinner('Listening...'):
            print("Listening...")
            r.pause_threshold = 1
            audio = r.listen(source)
    try:
        with st.spinner('Recognising...'):
            print("Recognising...")
            query = r.recognize_google(audio, language='en-in')
            print(f"User said: {query}\n")

    except Exception as e:
        print(e)
        print("Say that again please...")  
        return None

    return query

def main():
    st.header("Indian Sign Language Translation using Human-Computer Interaction")
    
    option1 = "Sign Language to Text/Speech"
    option2 = "Text/Speech to Sign Language"

    st.sidebar.title("Options")

    app_mode = st.sidebar.selectbox( "Choose the app mode",
        [
            option1,
            option2
        ],
    )

    if app_mode == option1:
        sign_language_detector()
    elif app_mode == option2:
        text_speech_to_sign()

def signGenerator(statement):
    words = statement.split()
    words = [word.upper() for word in words if word not in '.,!?:;']
    for word in words:
        if word.lower() in ['little', 'hi']:
            st.write(word.upper())
            st.image('./signs/{}.jpg'.format(word.capitalize()), width=75)
        else:
            imageList = []
            for letter in word:
                if letter not in ',.!?;:()':
                    imageList.append(cv2.resize(cv2.imread('./signs/{}.jpg'.format(letter)), (75, 75)))

            st.write(word)
            st.image(cvzone.stackImages(imageList, 8 if len(imageList) >= 8 else len(imageList), 1), width=75*(8 if len(imageList) >=8 else len(imageList)))

def text_speech_to_sign():

    st.markdown(f'<h1 style="color:#33ff33; font-size:24px;">{"Text/Speech to Sign Language"}</h1>', unsafe_allow_html=True)

    option = st.selectbox(
     'How would you like to convey the message?',
     ('Text to Sign Language', 'Speech to Sign Language'))

    st.write('You selected:', option)

    if option == 'Text to Sign Language':
        text = st.text_area("Enter the text to be translated", "")
        text = text.lower()

        if st.button("Translate", key='translate'):
            # if 'i love you' in text:
            #     text = text.replace('i love you', '')
            #     st.write('I LOVE YOU')
            #     st.image('./signs/I Love You.jpg', width=75)

            signGenerator(text)

            # try:
            #     image = Image.open('./signs/{}.jpg'.format(text))
            #     st.success(f"Translated text is: ")
            #     st.image(image, caption='Predicted sign')
            # except:
            #     st.error("No sign recognised")
    else:
        if st.button('Record', key='record'):
            query = takeCommand()

            if query is None:
                st.error("Speech recognition failed, Please try again")
            else:
                st.write("You said: ")
                st.success(query)
                signGenerator(query)

def sign_language_detector():
    st.markdown(f'<h1 style="color:#33ff33; font-size:24px;">{"Sign Language to Text/Speech"}</h1>', unsafe_allow_html=True)
    class OpenCVVideoProcessor(VideoProcessorBase):

        def __init__(self):
            self.detector = HandDetector(detectionCon=0.8, maxHands=2)
            self.ROI_top = 100
            self.ROI_bottom = 300
            self.ROI_right = 150
            self.ROI_left = 350
            self.background = None
            self.accumulated_weight = 0.5
            self.model = keras.models.load_model("signModelNew")

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

            # while True:
            img = cv2.flip(img,1)
            hands, img = self.detector.findHands(img)  # with draw

            if hands:
                hand1 = hands[0]
                bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h

                fingers1 = self.detector.fingersUp(hand1)

                self.ROI_top = bbox1[1] - 20
                self.ROI_bottom = bbox1[3] + bbox1[1] + 20
                self.ROI_right =  bbox1[0] - 20
                self.ROI_left = bbox1[2] + bbox1[0] + 20

                if len(hands) == 2:
                    hand2 = hands[1]
                    bbox2 = hand2["bbox"]  # Bounding box info x,y,w,h
                    handType2 = hand2["type"]

                    fingers2 = self.detector.fingersUp(hand2)

                    absDiff = abs((bbox1[1] + bbox1[3]) - (bbox2[1] + bbox2[3]))

                    self.ROI_top = bbox1[1] - 20 if bbox1[1] < bbox2[1] else bbox2[1] - 20
                    self.ROI_bottom = bbox1[3] + bbox1[1] + 20 + absDiff if bbox1[1] < bbox2[1] else bbox2[3] + bbox2[1] + 20 + absDiff
                    self.ROI_right =  bbox1[0] - 20 if handType2 == 'Left' else bbox2[0] - 20
                    self.ROI_left = bbox1[2] + (bbox1[0] if handType2 == 'Left' else bbox2[0]) + bbox2[2] + 80

            # roi = img[self.ROI_top:self.ROI_bottom, self.ROI_right:self.ROI_left]
            roi = img[self.ROI_top if self.ROI_top >= 0 else 0 :self.ROI_bottom, self.ROI_right if self.ROI_right >= 0 else 0:self.ROI_left]

            if not hands:
                cv2.putText(img, "No hands detected", (170, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)

            else:
                hand = self.segment_hand(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                if hand is not None:
                    thresholded = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    (_, thresholded) = cv2.threshold(thresholded, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    
                    thresholded = cv2.resize(thresholded, (64, 64))
                    thresholded = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)
                    thresholded = np.reshape(thresholded, (1,thresholded.shape[0],thresholded.shape[1],3))

                    pred = self.model.predict(thresholded)
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