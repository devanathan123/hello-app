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

Products_added = []
out_line=[]
in_line=[]
Final=[]
U_Final=[]
current_total =0
Free = []


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
    
    # -----Background Subtractor---------------------------------------
    backgroundObject = cv2.createBackgroundSubtractorMOG2(history=2)
    kernel = np.ones((3, 3), np.uint8)

    cap_s.set(3, 1920)
    cap_s.set(4, 1080)

    cap_t.set(3, 1920)
    cap_t.set(4, 1080)

    image_width = int(cap_s.get(3))
    image_height = int(cap_s.get(4))

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

    
    #------ BOUNDARY LINES --------------------------------------------------------------
    left_limits1 = [250 , 50 ,250, 1000 ]
    left_limits2 = [350 , 50, 350, 1000 ]

    right_limits1 = [1650 ,50, 1650 , 1000 ]
    right_limits2 = [1550 ,50, 1550 , 1000 ]

    top_limits1 = [250 , 50 , 1650 , 50 ]
    top_limits2 = [250 , 100 , 1550 , 100]

    bottom_limits1 = [250 ,1000 , 1650 ,1000]
    bottom_limits2 = [350 , 950 , 1550 , 950]

    #-------SIDE Window-----------------------------------------------------------------------------------
    top_limits1_s = [0 , 450 , 1920 ,450 ]
    top_limits2_s = [0 , 500 , 1920 ,500 ]
    top_limits3_s = [0 , 650 , 1920 ,650 ]


    #!!!!!!! -details !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    tb="shop3"
    product="Mango"
    db=firestore.client()
    doc_ref=db.collection(tb).document(product)
    doc=doc_ref.get()
    if doc.exists:
      #st.title(doc.to_dict())
      doc_data = doc.to_dict()
      field_value = doc_data.get('Stock')
      kpi4_text.write(f"<h1  style='color:red;'>{field_value}</h1>",unsafe_allow_html=True)    
      #st.title(field_value)
   
    else:
      st.title("NOT FOUND")


    current_total=0

    # ----SIDE-----------
    totalCount_s = []

    Total_products_s = 0
    Products_added_s = []
    Products_removed_s = []
    out_line_s = []
    in_line_s = []

    # ---TOP----------------
    totalCount_t = []

    Total_products_t = 0
    Products_added_t = []
    Products_removed_t = []

    out_line_t = []
    in_line_t = []

    #Final = []
    Hide = []
    Hide_remove = []
    Segment_remove = []

    # -----------Directions------------
    Left = []
    Right = []
    Top = []
    Bottom = []

    # ----Time--------------
    start = time.time()
    Hide_add_time = 0
    Hide_remove_time = 0

    
    while cap_t.isOpened() and cap_s.isOpened() and not stop_button:
              success_t, img_t= cap_t.read()
              success_s, img_s= cap_s.read()
              results_t = model(img_t, stream=True)
              results_s = model(img_s, stream=True)
              detections_t = np.empty((0,5))
              detections_s = np.empty((0,5))
              allArray_t = []
              allArray_s = []
              currentClass_s = ""
              currentClass_t = ""

              if success_t:
                  res_t = model.track(img_t, conf=0.3, persist=True, tracker="botsort.yaml")
                  res_plotted_t = res_t[0].plot()
                   
                  stframe_t.image(res_plotted_t,#caption='Detected Video',
                                      channels="BGR",use_column_width=True)

              if success_s:
                  res_s = model.track(img_s, conf=0.3, persist=True, tracker="botsort.yaml")
                  res_plotted_s = res_s[0].plot()
                   
                  stframe_s.image(res_plotted_s,#caption='Detected Video',
                                      channels="BGR",use_column_width=True)
    
    st.title("!! DONE !!")
