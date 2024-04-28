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
from collections import defaultdict
from ultralytics.utils.plotting import Annotator
from sort import *

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


    
        
     
#      kpi4_text.write(f"<h1  style='color:red;'>{value_to_select}</h1>",unsafe_allow_html=True)
#      kpi3_text.write(f"<h1  style='color:red;'>{row[0]}</h1>",unsafe_allow_html=True)
#      kpi2_text.write(f"<h1 style='color:red;'>{'{:.1f}'.format(row[1])}</h1>",unsafe_allow_html=True)
#      if len(Free_tmp)>0 and ele==Discount_Product:
#        Free_tmp.remove(Discount_Product)
#      else:
#        current_total = current_total + row[1]
#        U_Final.append(value_to_select)
#      kpi1_text.write(f"<h1 style='color:white;'>{'{:.1f}'.format(current_total)}</h1>",unsafe_allow_html=True)


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

    #--Tracking------------------------
    # tracker_s = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    # tracker_t = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    
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

              names = model.model.names


              if success_t:
                # results = model.track(img_t, persist=True,
                #               tracker="bytetrack.yaml")

                # boxes = results[0].boxes.xywh.cpu()
                # clss = results[0].boxes.cls.cpu().tolist()
                # track_ids = results[0].boxes.id.int().cpu().tolist()
        
                # annotator = Annotator(img_t, line_width=2,
                #                       example=str(names))
        
                # for box, track_id, cls in zip(boxes, track_ids, clss):
                #     x, y, w, h = box
                #     x1, y1, x2, y2 = (x - w / 2, y - h / 2,
                #                       x + w / 2, y + h / 2)
                #     label = str(names[cls]) + " : " + str(track_id)
                #     annotator.box_label([x1, y1, x2, y2],
                #                         label, (218, 100, 255))
        
                #     # Tracking Lines plot
                #     track = track_history[track_id]
                #     track.append((float(box[0]), float(box[1])))
                #     if len(track) > 30:
                #         track.pop(0)
        
                #     points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                #     cv2.polylines(img_t, [points], isClosed=False,
                #                   color=(37, 255, 225), thickness=2)
        
                #     # Center circle
                #     cv2.circle(img_t,
                #                (int(track[-1][0]), int(track[-1][1])),
                #                5, (235, 219, 11), -1)

                #     # res_t = model.track(img_t, conf=0.3, persist=True, tracker="botsort.yaml")
                #     # #res_plotted_t = res_t[0].plot()
                #     # #st.title(res_t[0])
    
                #     # stframe_t.image(res_plotted_t,#caption='Detected Video',
                #     #                   channels="BGR",use_column_width=True)
                      
                      
                # cv2.line(img_t, (left_limits1[0], left_limits1[1]), (left_limits1[2], left_limits1[3]), (0, 0, 255), 3)
                # cv2.line(img_t, (left_limits2[0], left_limits2[1]), (left_limits2[2], left_limits2[3]), (255, 0, 0), 3)
    
                # cv2.line(img_t, (right_limits1[0], right_limits1[1]), (right_limits1[2], right_limits1[3]), (0, 0, 255), 3)
                # cv2.line(img_t, (right_limits2[0], right_limits2[1]), (right_limits2[2], right_limits2[3]), (255, 0, 0), 3)
    
                # cv2.line(img_t, (top_limits1[0], top_limits1[1]), (top_limits1[2], top_limits1[3]), (0, 0, 255), 3)
                # cv2.line(img_t, (top_limits2[0], top_limits2[1]), (top_limits2[2], top_limits2[3]), (255, 0, 0), 3)
    
                # cv2.line(img_t, (bottom_limits1[0], bottom_limits1[1]), (bottom_limits1[2], bottom_limits1[3]), (0, 0, 255), 3)
                # cv2.line(img_t, (bottom_limits2[0], bottom_limits2[1]), (bottom_limits2[2], bottom_limits2[3]), (255, 0, 0), 3)

                # stframe_t.image(img_t, channels='BGR', use_column_width=True)


                  for r in results_t:
                      boxes = r.boxes
  
                      for box in boxes:
                          # Bounding Box
                          x1, y1, x2, y2 = box.xyxy[0]
                          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
  
                          w, h = x2 - x1, y2 - y1
                          cx, cy = x1 + w // 2, y1 + h // 2
                          cv2.circle(img_t, (cx, cy), 7, (0, 0, 255), cv2.FILLED)
                          # cv2.rectangle(img_s, (x1, y1), (x2, y2), (0, 0, 255), 2)
                          # Confidence
                          conf = math.ceil((box.conf[0] * 100)) / 100
                          # Class Name
                          cls = int(box.cls[0])
  
                          currentClass_t = classNames[cls]
  
                          if currentClass_t != "person" and conf > 0.3 and 650 > cy:
  
                              cvzone.putTextRect(img_t, f'{currentClass_t} {conf}',(max(0, x1), max(35, y1)), scale=3, thickness=3)  # Class Name
                              cv2.rectangle(img_t, (x1, y1), (x2, y2), (0, 255, 0), 2)
                              allArray_t.append([x1, y1, x2, y2, currentClass_t])
                              currentArray_t = np.array([x1, y1, x2, y2, conf])
                              detections_t = np.vstack((detections_t, currentArray_t))

                  cv2.line(img_t, (left_limits1[0], left_limits1[1]), (left_limits1[2], left_limits1[3]), (0, 0, 255), 3)
                  cv2.line(img_t, (left_limits2[0], left_limits2[1]), (left_limits2[2], left_limits2[3]), (255, 0, 0), 3)

                  cv2.line(img_t, (right_limits1[0], right_limits1[1]), (right_limits1[2], right_limits1[3]), (0, 0, 255), 3)
                  cv2.line(img_t, (right_limits2[0], right_limits2[1]), (right_limits2[2], right_limits2[3]), (255, 0, 0), 3)

                  cv2.line(img_t, (top_limits1[0], top_limits1[1]), (top_limits1[2], top_limits1[3]), (0, 0, 255), 3)
                  cv2.line(img_t, (top_limits2[0], top_limits2[1]), (top_limits2[2], top_limits2[3]), (255, 0, 0), 3)

                  cv2.line(img_t, (bottom_limits1[0], bottom_limits1[1]), (bottom_limits1[2], bottom_limits1[3]), (0, 0, 255), 3)
                  cv2.line(img_t, (bottom_limits2[0], bottom_limits2[1]), (bottom_limits2[2], bottom_limits2[3]), (255, 0, 0), 3)
  
                  stframe_t.image(img_t, channels='BGR', use_column_width=True)
  
              else:
                  break

              if success_s:
                # Tracking New MODEL
                # res_s = model.track(img_s, conf=0.3, persist=True, tracker="botsort.yaml")
                # res_plotted_s = res_s[0].plot()
                
                # stframe_s.image(res_plotted_s,
                #    #caption='Detected Video',
                #    channels="BGR",use_column_width=True)
                  
                # cv2.line(img_s, (top_limits1_s[0], top_limits1_s[1]), (top_limits1_s[2], top_limits1_s[3]), (0, 0, 255), 3)
                # cv2.line(img_s, (top_limits2_s[0], top_limits2_s[1]), (top_limits2_s[2], top_limits2_s[3]), (255, 0, 0), 3)
                # cv2.line(img_s, (top_limits3_s[0], top_limits3_s[1]), (top_limits3_s[2], top_limits3_s[3]), (255, 0, 0), 3)
                
                # #stframe_s.image(img_s, channels='BGR', use_column_width=True)

  
                  for r in results_s:
                      boxes = r.boxes
  
                      for box in boxes:
                          # Bounding Box
                          x1, y1, x2, y2 = box.xyxy[0]
                          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
  
                          w, h = x2 - x1, y2 - y1
                          cx, cy = x1 + w // 2, y1 + h // 2
                          cv2.circle(img_s, (cx, cy), 7, (0, 0, 255), cv2.FILLED)
  
                          # cv2.rectangle(img_s, (x1, y1), (x2, y2), (0, 0, 255), 2)
                          # Confidence
                          conf = math.ceil((box.conf[0] * 100)) / 100
                          # Class Name
                          cls = int(box.cls[0])
  
                          currentClass_s = classNames[cls]
  
                          if currentClass_s != "person" and conf > 0.3 and 650 > cy:
  
                              cvzone.putTextRect(img_s, f'{currentClass_s} {conf}',(max(0, x1), max(35, y1)),scale=3, thickness=3)  # Class Name
                              cv2.rectangle(img_s, (x1, y1), (x2, y2), (0, 255, 0), 2)
                              allArray_s.append([x1, y1, x2, y2, currentClass_s])
                              currentArray_s = np.array([x1, y1, x2, y2, conf])
                              detections_s = np.vstack((detections_s, currentArray_s))
                  
                  
                  cv2.line(img_s, (top_limits1_s[0], top_limits1_s[1]), (top_limits1_s[2], top_limits1_s[3]), (0, 0, 255), 3)
                  cv2.line(img_s, (top_limits2_s[0], top_limits2_s[1]), (top_limits2_s[2], top_limits2_s[3]), (255, 0, 0), 3)
                  cv2.line(img_s, (top_limits3_s[0], top_limits3_s[1]), (top_limits3_s[2], top_limits3_s[3]), (255, 0, 0), 3)
                  
                  stframe_s.image(img_s, channels='BGR', use_column_width=True)
  
              else:
                  break
        
              resultsTracker_s = tracker_s.update(detections_s)
              resultsTracker_t = tracker_t.update(detections_t)

              for result in resultsTracker_t:

                x1, y1, x2, y2, id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))

                cx, cy = x1 + w // 2, y1 + h // 2
                cv2.circle(img_t, (cx, cy), 7, (0, 0, 255), cv2.FILLED)

                # -------To Get the CurrentClass for the Objects detected--------------------------------
                for r in allArray_t:
                    if (r[0] - 50 < x1 < r[0] + 50 and r[1] - 50 < y1 < r[1] + 50 and r[2] - 50 < x2 < r[2] + 50 and r[3] - 50 < y2 < r[3] + 50):
                        currentClass_t = r[4]

                # -------------- Bounding Box for the objects inside the CART ----------------------------
                if left_limits1[0] < cx < right_limits1[0] and top_limits1[1] < cy < bottom_limits1[1]:
                    cvzone.putTextRect(img_t, f' {int(id)}', (max(0, cx), max(35, cy)), scale=1, thickness=1)
                    cv2.rectangle(img_t, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # left-LINE---------------------------------------------------------------------------
                # ------LEFT OUTER LIMIT-----------------------------------------
                if left_limits1[0] - 25 < cx < left_limits1[2] + 25 and left_limits1[1] < cy < left_limits1[3]:
                    cv2.line(img_t, (left_limits1[0], left_limits1[1]), (left_limits1[2], left_limits1[3]), (0, 255, 0), 5)

                    if out_line_t.count(id) == 0 and in_line_t.count(id) == 0:
                        # if totalCount.count(id) == 0:
                        out_line_t.append(id)
                        print("out-1")

                    else:
                        # ---------- REMOVE ITEM ------------------

                        if out_line_t.count(id) == 0 and in_line_t.count(id) == 1:
                            print("out-2")
                            # totalCount.remove(id)
                            Total_products_t = Total_products_t - 1
                            # print(classArray[cnt])
                            in_line_t.remove(id)

                # -------LEFT INNER LIMIT------------------------------------------------------

                if left_limits2[0] - 25 < cx < left_limits2[2] + 25 and left_limits2[1] < cy < left_limits2[3]:
                    cv2.line(img_t, (left_limits2[0], left_limits2[1]), (left_limits2[2], left_limits2[3]),(0, 255, 0), 5)
                    if in_line_t.count(id) == 0 and out_line_t.count(id) == 0:
                            # if totalCount.count(id) == 0:
                        print("in-1")
                        in_line_t.append(id)
                    else:
                        # ------------ ADD ITEM ----------------------
                        if in_line_t.count(id) == 0 and out_line_t.count(id) == 1:

                            print("in-2")
                            # totalCount.remove(id)
                            Total_products_t = Total_products_t + 1
                            # print(classArray[cnt])
                            out_line_t.remove(id)
    
                # right----------------------------------------------
                # ------RIGHT OUTER LIMIT-----------------------------------------
                if right_limits1[0] + 25 > cx > right_limits1[2] - 25 and right_limits1[1] < cy < right_limits1[3]:
                    cv2.line(img_t, (right_limits1[0], right_limits1[1]), (right_limits1[2], right_limits1[3]),(0, 255, 0),5)
                    if out_line_t.count(id) == 0 and in_line_t.count(id) == 0:
                            # if totalCount.count(id) == 0:
                        out_line_t.append(id)
                        print("out-1-right")

                    else:
                        # ---------- REMOVE ITEM ------------------
                        if out_line_t.count(id) == 0 and in_line_t.count(id) == 1:
                            print("out-2")
                            # totalCount.remove(id)
                            Total_products_t = Total_products_t - 1
                            # print(classArray[cnt])
                            in_line_t.remove(id)
                

                # --------RIGHT INNER LIMIT ---------------------------------------
                if right_limits2[0] + 25 > cx > right_limits2[2] - 25 and right_limits2[1] < cy < right_limits2[3]:
                    cv2.line(img_t, (right_limits2[0], right_limits2[1]), (right_limits2[2], right_limits2[3]),(0, 255, 0),5)
                    if in_line_t.count(id) == 0 and out_line_t.count(id) == 0:
                        # if totalCount.count(id) == 0:
                        print("in-1")
                        in_line_t.append(id)
                    else:
                        # ------------ ADD ITEM ----------------------
                        if in_line_t.count(id) == 0 and out_line_t.count(id) == 1:
                            print("in-2")
                            # totalCount.remove(id)
                            Total_products_t = Total_products_t + 1
                            # print(classArray[cnt])
                            out_line_t.remove(id)

                # ----- TOP----------------------------------------------------------------------------------------------------------
                # --------------------TOP-OUTER LINE-----------------------------------------

                if top_limits1[0] < cx < top_limits1[2] and top_limits1[1] - 25 < cy < top_limits1[3] + 25:
                    cv2.line(img_t, (top_limits1[0], top_limits1[1]), (top_limits1[2], top_limits1[3]), (0, 255, 0),5)
                    if out_line_t.count(id) == 0 and in_line_t.count(id) == 0:
                            # if totalCount.count(id) == 0:
                        out_line_t.append(id)
                        print("out-1")

                    else:
                        # ---------- REMOVE ITEM ------------------
                        if out_line_t.count(id) == 0 and in_line_t.count(id) == 1:
                            print("out-2")
                            # totalCount.remove(id)
                            Total_products_t = Total_products_t - 1
                            # print(classArray[cnt])
                            in_line_t.remove(id)

                # -------TOP INNER LIMIT------------------------------------------------------
                if top_limits2[0]  < cx < top_limits2[2]  and top_limits2[1] - 25 < cy < top_limits2[3] + 25:
                    cv2.line(img_t, (top_limits2[0], top_limits2[1]), (top_limits2[2], top_limits2[3]),(0, 255, 0), 5)
                    if in_line_t.count(id) == 0 and out_line_t.count(id) == 0:
                        # if totalCount.count(id) == 0:
                        print("in-1")
                        in_line_t.append(id)
                    else:
                        # ------------ ADD ITEM ----------------------
                        if in_line_t.count(id) == 0 and out_line_t.count(id) == 1:
                            print("in-2")
                            # totalCount.remove(id)
                            Total_products_t = Total_products_t + 1
                            # print(classArray[cnt])
                            out_line_t.remove(id)

                # -----BOTTOM -------------------------------------------
                # ---------------------BOTTOM OUTER LIMIT------------------------------------------------------------
                if bottom_limits1[0] < cx < bottom_limits1[2] and bottom_limits1[1] - 25 < cy < bottom_limits1[3] + 25:
                    cv2.line(img_t, (bottom_limits1[0], bottom_limits1[1]), (bottom_limits1[2], bottom_limits1[3]),
                                 (0, 255, 0), 5)
                    if out_line_t.count(id) == 0 and in_line_t.count(id) == 0:
                        # if totalCount.count(id) == 0:
                        out_line_t.append(id)
                        print("out-1")

                    else:
                        # ---------- REMOVE ITEM ------------------
                        if out_line_t.count(id) == 0 and in_line_t.count(id) == 1:
                            print("out-2")
                            # totalCount.remove(id)
                            Total_products_t = Total_products_t - 1
                            # print(classArray[cnt])
                            in_line_t.remove(id)

                # -------BOTTOM INNER LIMIT------------------------------------------------------
                if bottom_limits2[0] - 25 < cx < bottom_limits2[2] + 25 and bottom_limits2[1] < cy < bottom_limits2[3]:
                    cv2.line(img_t, (bottom_limits2[0], bottom_limits2[1]), (bottom_limits2[2], bottom_limits2[3]),(0, 255, 0), 5)
                    if in_line_t.count(id) == 0 and out_line_t.count(id) == 0:
                        # if totalCount.count(id) == 0:
                        print("in-1")
                        in_line_t.append(id)
                    else:
                        # ------------ ADD ITEM ----------------------
                        if in_line_t.count(id) == 0 and out_line_t.count(id) == 1:
                            print("in-2")
                            # totalCount.remove(id)
                            Total_products_t = Total_products_t + 1
                            # print(classArray[cnt])
                            out_line_t.remove(id)
    
    

    
    st.title("!!FINISHED !!")
