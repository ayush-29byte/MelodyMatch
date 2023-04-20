#import necessary libraries
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import webbrowser

#Load emotion detection and model and lables
model = load_model("model.h5")
label = np.load("labels.npy")
# Load holistic and hand tracking models from mediapipe library
holistic = mp.solutions.holistic
hands = mp.solutions.hands
# Create instance of holistic model
holis = holistic.Holistic()
# Load drawing utilities from mediapipe
drawing = mp.solutions.drawing_utils

st.header("MelodyMatch")

if "run" not in st.session_state:
    st.session_state["run"] = "true"
# Try to load emotion from previously stored file, else set emotion to empty string
try:
    emotion = np.load("emotion.npy")[0]
except:
    emotion = ""

if not(emotion):
    st.session_state["run"] = "true"
else:
    st.session_state["run"] = "false"

# Create class to process emotions from webcam frames
class ProcessEmotion:
    def recv(self,frame):
        frm = frame.to_ndarray(format="bgr24")
        ######################################

        # Flip the frame horizontally for better visualization
        frm = cv2.flip(frm, 1)

        # Process the frame using the holistic model to detect landmarks on face and body

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        #creating list to store the landmarks
        lst = []

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                for i in range(42):
                    lst.append(0.0)

            lst = np.array(lst).reshape(1, -1)

            pred = label[np.argmax(model.predict(lst))]

            print(pred)
            cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

            np.save("emotion.npy",np.array([pred]))

        drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
        drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
        drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)
        ######################################
        return av.VideoFrame.from_ndarray(frm,format="bgr24")

language = st.text_input("Enter Your Preferred Melody Language")
singer = st.text_input("Let Me Know Your Favorite Singer")

#to acess the webcamera for emotion sensing
if language and singer and st.session_state["run"] != "false":
    webrtc_streamer(key="key",desired_playing_state=True,video_processor_factory=ProcessEmotion)

btn = st.button("Match To My Melody")

if btn:
    if not(emotion):
        st.warning("Please Let Me Read Your Mood First")
        st.session_state["run"] = "true"
    else:
        webbrowser.open(f"https://www.youtube.com/results?search_query={language}+{emotion}+song+{singer}")
        np.save("emotion.npy",np.array([""]))
        st.session_state["run"] = "false"