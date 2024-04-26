import streamlit as st
import os
import opencv
import numpy as np
from ultralytics import YOLO
import cvzone
import math
from pathlib import Path
import sys

def main():
    #st.markdown("<center>welcome</center>",unsafe_allow_html=True,)

    #-------------HOME PAGE---------------------------------

    st.header('WELCOME TO e-SHOPPING CART', divider='rainbow')
    #st.markdown("</center>",unsafe_allow_html=True)
    st.title("_SHOP XYZ_")
    st.sidebar.header("Settings",divider='rainbow')
    st.sidebar.subheader("Parameters")

    app_mode = st.sidebar.selectbox('Choose the Mode', ['HOME','Product Counter'])

    if app_mode == 'HOME':
        st.markdown(':blue[This project uses **YOLO** for Object Detection on Images and Videos and we are using **StreamLit** to create a Graphical User Interface (GUI)]')
        st.image('https://play-lh.googleusercontent.com/z2pE7U4gpS3A4QKDMaMGqJTHFcQ_-rZMkjQ7IHYJk2gHONJg1xQJP-HAwGwBLbE1Exs')

    #---------Product Counter Model---------------------------

    elif app_mode == 'Product Counter':

        #--------------model---------------------
          # Get the absolute path of the current file
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
        #-----------------------------------------
      
        st.sidebar.markdown('---')
        use_webcam = st.sidebar.checkbox('Use Webcam')
        st.sidebar.markdown('---')

        uploaded_file_top = st.file_uploader("Upload a TOP-view video", type=["mp4", "avi"])
        uploaded_file_side = st.file_uploader("Upload a SIDE-view video", type=["mp4", "avi"])

        if uploaded_file_top and uploaded_file_side is not None:
          # Save the uploaded file to a temporary location
          with open("temp_video1.mp4", "wb") as temp_file_top:
            temp_file_top.write(uploaded_file_top.read())

          with open("temp_video2.mp4", "wb") as temp_file_side:
            temp_file_side.write(uploaded_file_side.read())
            
          # Display the video frame by frame
          stop_button = st.button("Stop")
          stframe_t = st.empty()
          cap_t = cv2.VideoCapture(temp_file_top.name)
          cap_s = cv2.VideoCapture(temp_file_side.name)
  
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
  
                  stframe.image(img_t, channels='BGR', use_column_width=True)
  
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
  
                  stframe.image(img_s, channels='BGR', use_column_width=True)
  
              else:
                  break

  
          st.title("DONE !!")

          
        
      #------------ OLD -------------------------
        # video_file_buffer = st.sidebar.file_uploader("Upload a Video", type = ["mp4", "avi", "mov", "asf"])

        # #----Default FILE PATH-------------
        # DEMO_VIDEO_S = 'Videos/t-13/input_side.mp4'
        # DEMO_VIDEO_T = 'Videos/t-13/input_top.mp4'

        # tffile = tempfile.NamedTemporaryFile(suffix= '.mp4', delete=False)

        # if not video_file_buffer:
        #     if use_webcam:
        #         tffile.name = 0
        #     else:
        #         vid_S = cv2.VideoCapture(DEMO_VIDEO_S)
        #         vid_T = cv2.VideoCapture(DEMO_VIDEO_T)

        #         tffile.name_S = DEMO_VIDEO_S
        #         demo_vid_S = open(tffile.name_S, 'rb')
        #         demo_bytes_S = demo_vid_S.read()
        #         st.sidebar.success("Running Demo")
        #         st.sidebar.text('Input Video')
        #         st.sidebar.video(demo_bytes_S)

        #         tffile.name_T = DEMO_VIDEO_T
        #         demo_vid_T = open(tffile.name_T, 'rb')
        #         demo_bytes_T = demo_vid_T.read()
        #         st.sidebar.video(demo_bytes_T)

        # else:
        #     tffile.write(video_file_buffer.read())
        #     demo_vid = open(tffile.name, 'rb')
        #     demo_bytes = demo_vid.read()
        #     st.sidebar.text('Input Video')
        #     st.sidebar.video(demo_bytes)

        # stframe_s = st.empty()
        # stframe_t = st.empty()

