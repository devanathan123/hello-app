import streamlit as st
import os
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
from pathlib import Path
import sys
import firebase_admin
from firebase_admin import credentials,firestore,auth
from google.cloud.firestore_v1.base_query import FieldFilter,Or
import requests
import time
#from sort import *

# Load JSON key file from GitHub
def load_json_key():
    url = "https://raw.githubusercontent.com/devanathan123/stream_app/main/private_key.json"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP errors (e.g., 404, 500)
        key_data = response.json()
        return key_data
    except Exception as e:
        st.error(f"Error fetching JSON key: {e}")
        return None

# Check if Firebase Admin SDK is already initialized
if not firebase_admin._apps:
    # Initialize Firebase Admin SDK
    key_data = load_json_key()
    cred = credentials.Certificate(key_data)
    firebase_admin.initialize_app(cred)



def load_product_counter(video_name_s,video_name_t, kpi1_text, kpi2_text, kpi3_text, kpi4_text,kpi5_text,stframe_s,stframe_t):
    cap_s = cv2.VideoCapture(video_name_s)
    cap_t = cv2.VideoCapture(video_name_t)
    cap_s.set(3, 1920)
    cap_s.set(4, 1080)

    cap_t.set(3, 1920)
    cap_t.set(4, 1080)
   #----- MODEL ----------------------------------------------------------
    #Get the absolute path of the current file
    FILE = Path(__file__).resolve()
    # Get the parent directory of the current file
    ROOT = FILE.parent
    # Add the root path to the sys.path list if it is not already there
    if ROOT not in sys.path:
        sys.path.append(str(ROOT))
    # Get the relative path of the root directory with respect to the current working directory
    ROOT = ROOT.relative_to(Path.cwd())

    MODEL_DIR = ROOT / 'YOLO-Weights'
    DETECTION_MODEL = MODEL_DIR / 'seg3n_25.pt'
    model = YOLO(DETECTION_MODEL)
    classNames = ['Cinthol_Soap', 'Hamam_Soap', 'Him_Face_Wash', 'Maa_Juice', 'Mango', 'Mysore_Sandal_Soap','Patanjali_Dant_Kanti', 'Tide_Bar_Soap', 'ujala_liquid']
    #st.title("Streamlit Video Player")
    
    #Display the video frame by frame
    stop_button = st.button("Stop")
    
    while cap_t.isOpened() and cap_s.isOpened() and not stop_button:
              success_t, img_t= cap_t.read()
              success_s, img_s= cap_s.read()

              if success_t:
                    res_t = model.track(img_t, conf=0.3, persist=True, tracker="botsort.yaml")
                    res_plotted_t = res_t[0].plot()
                    stframe_t.image(res_plotted_t,channels="BGR",use_column_width=True)

        
              if success_s:
                  res_s = model.track(img_s, conf=0.3, persist=True, tracker="botsort.yaml")
                  res_plotted_s = res_s[0].plot()
                  stframe_s.image(res_plotted_s,channels="BGR",use_column_width=True)
    
    st.title("!! DONE !!")
