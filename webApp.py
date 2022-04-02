import keras
from HandTrackingModule import HandDetector
import cvzone

import av
import cv2

import numpy as np
import streamlit as st
import pydub
import queue

import speech_recognition as sr

import pyttsx3

#audio of system to respond
# engine = pyttsx3.init('sapi5')
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[0].id)
# engine.setProperty('rate',180)

from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

lm_alpha = 0.931289039105002
lm_beta = 1.1834137581510284
beam = 100

previous_sign = ''

def speak(audio):
    engine.say(audio)
    # engine.runAndWait()

def app_stt():
    '''
    It takes the audio buffer, converts it to a wav file, and then uses Google's speech to text API to
    convert the audio to text.
    '''

    # st.write("testing")
    
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=2048,
        rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        media_stream_constraints={"video": False, "audio": True},
    )

    # st.write("testing 2")

    if "audio_buffer" not in st.session_state:
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

    status_indicator = st.empty()

    while True:
        # with st.spinner("Loading..."):
        #     pass
            # while webrtc_ctx.audio_receiver:
            #     pass

        # for percent_complete in range(100):
        #     time.sleep(0.1)
        #     my_bar.progress(percent_complete + 1)

        if webrtc_ctx.audio_receiver:

            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.success("Running. Say something!")

            sound_chunk = pydub.AudioSegment.empty()
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound

            if len(sound_chunk) > 0:
                st.session_state["audio_buffer"] += sound_chunk

        else:
            # status_indicator.write("AudioReciver is not set. Abort.")
            break

    audio_buffer = st.session_state["audio_buffer"]

    if not webrtc_ctx.state.playing and len(audio_buffer) > 0:
        # st.info("Writing wav to disk")
        # audio_buffer.export("temp.wav", format="wav")
        filename = "audio.wav"
        audio_buffer.export(filename, format ="wav")
        AUDIO_FILE = filename

        r = sr.Recognizer()
        with sr.AudioFile(AUDIO_FILE) as source:
            audio_listened = r.listen(source)

        try:
            text = r.recognize_google(audio_listened, language='en-in')
            # print(text)
            if text is None:
                st.error("Speech recognition failed, Please try again")
            else:
                st.write("You said: ")
                st.success(text)
                signGenerator(text)
        
        except sr.UnknownValueError:
            st.error("Speech recognition failed, Please try again")
            pass

        except sr.RequestError as e:
            print("Could not request results.")

        # Reset
        st.session_state["audio_buffer"] = pydub.AudioSegment.empty()

# def takeCommand():
#     r = sr.Recognizer()

#     audio = pyaudio.PyAudio()
#     # st.write(audio.get_default_input_device_info())

#     with sr.Microphone() as source:
    
#         r.adjust_for_ambient_noise(source, duration=2)
#         with st.spinner('Listening...'):
#             print("Listening...")
#             r.pause_threshold = 1
#             audio = r.listen(source)
#     try:
#         with st.spinner('Recognising...'):
#             print("Recognising...")
#             query = r.recognize_google(audio, language='en-in')
#             print(f"User said: {query}\n")

#     except Exception as e:
#         print(e)
#         print("Say that again please...")  
#         return None

#     return query

def main():
    st.header("Indian Sign Language Translation For Hard-of-Hearing and Hard-of-speaking Community")

    option1 = "Sign Language to Text/Speech"
    option2 = "Text/Speech to Sign Language"

    st.sidebar.image("logo.png")
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

units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]

tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

scales = ["hundred", "thousand", "million", "billion", "trillion"]

def text2int(textnum, numwords={}):

    if not numwords:
      

      numwords["and"] = (1, 0)
      for idx, word in enumerate(units):    numwords[word] = (1, idx)
      for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
      for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    for word in textnum.split():
        if word not in numwords:
          raise Exception("Illegal word: " + word)

        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    return result + current

def signGenerator(statement):

    '''
    It takes in a string, splits it into words, and then for each word, it checks if it is a number, a
    unit, or a tens word. If it is, it converts it to a number and then converts that number to a list
    of images. If it is not, it converts the word to a list of images. Then, it stacks the images into a
    single image and displays it.
    
    :param statement: The statement to be converted to a sign
    '''

    words = statement.split()
    words = [word.upper() for word in words if word not in '.,!?:;']

    # for word in words:

    for word in words:
        if word.lower() in ['little', 'hi']:
            st.write(word.upper())
            st.image('./signs/{}.jpg'.format(word.capitalize()), width=75)

        elif word.lower() in units:
            number = text2int(word.lower())
            imageList = []
            for letter in str(number):
                if letter not in ',.!?;:()':
                    imageList.append(cv2.resize(cv2.imread('./signs/{}.jpg'.format(letter)), (75, 75)))

            st.write(word)
            st.image(cvzone.stackImages(imageList, 8 if len(imageList) >= 8 else len(imageList), 1), width=75*(8 if len(imageList) >=8 else len(imageList)))

        else:
            imageList = []
            for letter in word:
                if letter not in ',.!?;:()':
                    imageList.append(cv2.resize(cv2.imread('./signs/{}.jpg'.format(letter)), (75, 75)))

            st.write(word)
            st.image(cvzone.stackImages(imageList, 8 if len(imageList) >= 8 else len(imageList), 1), width=75*(8 if len(imageList) >=8 else len(imageList)))

def text_speech_to_sign():
    '''
    It takes the text input from the user and converts it into a sign language image.
    '''

    st.markdown(f'<h1 style="color:#33ff33; font-size:24px;">{"Text/Speech to Sign Language"}</h1>', unsafe_allow_html=True)

    option = st.selectbox(
     'How would you like to convey the message?',
     ('Text to Sign Language', 'Speech to Sign Language'))

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
        # if st.button('Record', key='record'):

            # query = takeCommand()
        # with st.spinner('Loading...'):
        st.info('''Please wait till the microphone is initialised and ready message is displayed.
                   Once the microphone is ready, start speaking and then click on the button again to stop recording.''')
                   
        app_stt()

        # if query is None:
        #     st.error("Speech recognition failed, Please try again")
        # else:
        #     st.write("You said: ")
        #     st.success(query)
        #     signGenerator(query)

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
            global previous_sign
            img = frame.to_ndarray(format="bgr24")
            word_dict = {0:'One', 1:'Two', 2:'Three', 3:'I Love You', 4:'Little'}

            # while True:
            img = cv2.flip(img,1)
            hands, img = self.detector.findHands(img)  # with draw

            if hands:
                hand1 = hands[0]
                bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h

                # fingers1 = self.detector.fingersUp(hand1)

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
                    # if predText != previous_sign:
                    #     print("not same")
                    #     speak(predText)
                    #     previous_sign = predText
                        
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
    st.set_page_config(page_title="Signspeak", page_icon="signspeak.png")
    main()