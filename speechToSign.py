import speech_recognition as sr
import cv2

def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=2)
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source)
    try:
        print("Recognizing...")    
        query = r.recognize_google(audio, language='en-in')
        print(f"User said: {query}\n")

    except Exception as e:
        print("Say that again please...")  
        return "None"

    return query


if __name__ == "__main__":
    while True:
        print("Please speak out the sign")
        query = takeCommand().lower().split()[1]

        # query = input("Enter the number of sign: ")

        image = 'D:\\gesture\\train\\{}\\0.jpg'.format(query)
        if query != '':
            try:
                img = cv2.imread(image)
                cv2.imshow("Sign", img)
                cv2.waitKey(0)
            except:
                print("No sign recognised")
