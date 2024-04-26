import streamlit as st
import os
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
from pathlib import Path
import sys
import pyautogui

def load_product_counter(video_name_s,video_name_t, kpi1_text, kpi2_text, kpi3_text, kpi4_text,kpi5_text,stframe_s,stframe_t):
    cap_s = cv2.VideoCapture(video_name_s)
    cap_t = cv2.VideoCapture(video_name_t)
    cap_s.set(3, 1920)
    cap_s.set(4, 1080)

    cap_t.set(3, 1920)
    cap_t.set(4, 1080)

    image_width = int(cap_s.get(3))
    image_height = int(cap_s.get(4))

    screen_width, screen_height = pyautogui.size()
    
    frame_width = (image_width / screen_width)
    frame_height = (image_height / screen_height)

    device: str = "mps" if torch.backends.mps.is_available() else "cpu"

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
    st.title("Streamlit Video Player")
    
    #Display the video frame by frame
    stop_button = st.button("Stop")
    #--------------------------------------------------------------------------
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
  
                  for r in results_t:
                      boxes = r.boxes
  
                      for box in boxes:
                          # Bounding Box
                          x1, y1, x2, y2 = box.xyxy[0]
                          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
  
                          w, h = x2 - x1, y2 - y1
                          cx, cy = x1 + w // 2, y1 + h // 2
  
                          # cv2.rectangle(img_s, (x1, y1), (x2, y2), (0, 0, 255), 2)
                          # Confidence
                          conf = math.ceil((box.conf[0] * 100)) / 100
                          # Class Name
                          cls = int(box.cls[0])
  
                          currentClass_t = classNames[cls]
  
                          if currentClass_t != "person" and conf > 0.3 and 650 > cy:
  
                              cvzone.putTextRect(img_t, f'{currentClass_t} {conf}',
                                                 (max(0, x1), max(35, y1)),
                                                 scale=3, thickness=3)  # Class Name
                              cv2.rectangle(img_t, (x1, y1), (x2, y2), (0, 255, 0), 2)
  
                  stframe_t.image(img_t, channels='BGR', use_column_width=True)
  
              else:
                  break

              if success_s:
  
                  for r in results_s:
                      boxes = r.boxes
  
                      for box in boxes:
                          # Bounding Box
                          x1, y1, x2, y2 = box.xyxy[0]
                          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
  
                          w, h = x2 - x1, y2 - y1
                          cx, cy = x1 + w // 2, y1 + h // 2
  
                          # cv2.rectangle(img_s, (x1, y1), (x2, y2), (0, 0, 255), 2)
                          # Confidence
                          conf = math.ceil((box.conf[0] * 100)) / 100
                          # Class Name
                          cls = int(box.cls[0])
  
                          currentClass_s = classNames[cls]
  
                          if currentClass_t != "person" and conf > 0.3 and 650 > cy:
  
                              cvzone.putTextRect(img_s, f'{currentClass_s} {conf}',
                                                 (max(0, x1), max(35, y1)),
                                                 scale=3, thickness=3)  # Class Name
                              cv2.rectangle(img_s, (x1, y1), (x2, y2), (0, 255, 0), 2)
  
                  stframe_s.image(img_s, channels='BGR', use_column_width=True)
  
              else:
                  break

  
    st.title("DONE !!")
