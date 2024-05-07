#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
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
    left_limits1 = [250 , 100 ,250, 1000 ]
    left_limits2 = [350 , 100, 350, 1000 ]

    right_limits1 = [1650 ,100, 1650 , 1000 ]
    right_limits2 = [1550 ,100, 1550 , 1000 ]

    top_limits1 = [250 , 100 , 1650 , 100 ]
    top_limits2 = [250 , 200 , 1550 , 200]

    bottom_limits1 = [250 ,1000 , 1650 ,1000]
    bottom_limits2 = [350 , 950 , 1550 , 950]

    #-------SIDE Window-----------------------------------------------------------------------------------
    top_limits1_s = [0 , 450 , 1920 ,450 ]
    top_limits2_s = [0 , 500 , 1920 ,500 ]
    top_limits3_s = [0 , 650 , 1920 ,650 ]



    # #!!!!!!! -details !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # tb="shop3"
    # product="Mango"
    # db=firestore.client()
    # doc_ref=db.collection(tb).document(product)
    # doc=doc_ref.get()
    # if doc.exists:
    #   # To get data:-------------------------
    #   #st.title(doc.to_dict())
    #   #doc_data = doc.to_dict()
    #   #field_value = doc_data.get('Stock')
    #   #kpi4_text.write(f"<h1  style='color:red;'>{field_value}</h1>",unsafe_allow_html=True)    
    #   #st.title(field_value)

    #   # To Update data:----------------------
    #   #doc_ref.update({'Stock': firestore.Increment(1)})
    #   doc_ref.update({'Stock': firestore.Increment(-1)})
    #   doc=doc_ref.get()
    #   doc_data = doc.to_dict()
    #   field_value = doc_data.get('Stock')
    #   kpi4_text.write(f"<h1  style='color:red;'>{field_value}</h1>",unsafe_allow_html=True) 
   
    # else:
    #   st.title("NOT FOUND")


    tb="shop3"
    prod=["Him_Face_Wash","Maa_Juice"]
    db=firestore.client()
    for product in prod:
        doc_ref=db.collection(tb).document(product)
        doc=doc_ref.get()
        if doc.exists:
            doc_ref.update({'Stock':firestore.Increment(-1)})
            doc = doc_ref.get()
            doc_data = doc.to_dict()
            stock = doc_data.get('Stock')
            amt = doc_data.get('Amount')
            #det = doc_data.get('Category')
            det = product
            kpi4_text.write(f"<h1  style='color:red;'>{det}</h1>",unsafe_allow_html=True)
            kpi3_text.write(f"<h1  style='color:red;'>{stock}</h1>",unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='color:red;'>{amt}</h1>",unsafe_allow_html=True)
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

    # --- Mysql Connection -----------
    tb="shop3"
    db=firestore.client()

    # -- Discount Offer --------------
    Original_Product = "Hamam_Soap"
    Discount_Product = "Cinthol_Soap"

    #--Tracking------------------------
    tracker_s = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    tracker_t = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    
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

              #---------------DETECT-----------------------------------------

              movement_detector=[]
              # ------Time----------------
              fps = time.time()
              fps = int(fps - start)

              if fps - Hide_add_time > 50:
                  Hide = []

              if fps - Hide_remove_time > 50:
                  Hide_remove = []

              if success_t:
                  #---- Detect Moving object --------- https://learnopencv.com/moving-object-detection-with-opencv/
                  fgmask = backgroundObject.apply(img_t)
                  _, fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)
                  fgmask = cv2.erode(fgmask, kernel, iterations=1)
                  fgmask = cv2.dilate(fgmask, kernel, iterations=40)

                  # detect contour
                  countors, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                  forground = cv2.bitwise_and(img_t, img_t, mask=fgmask)

              else:
                  break

              #---SIDE VIEW-MASK-----------------------------------------------------
              if success_s:
                  # Run Yolov8 inference on the frame
                  results_s = model(img_s)
                  area_px_s = []
                  area_px_s_dummy = []
                  found_ele_s=[]

                  # visulaize the results on the frame without bounding box
                  #annotated_frame_s = results_s[0].plot(boxes=False)

                  # --- find mask -----------------------------------------
                  l = 0
                  masks_s = results_s[0].masks
                  if results_s[0].masks is not None:
                      l = len(masks_s)
                  print(l)
                  i = 0
                  while i < l:
                      mask1_s = masks_s[i]
                      mask_s = mask1_s.cpu().data[0].numpy()
                      polygon_s = mask1_s.xy[0]
                      shape_s = mask1_s.shape

                      mask_img_s = Image.fromarray(mask_s, "I")
                      pts_s = np.array([polygon_s], np.int32)
                      pts_s = pts_s.reshape((-1, 1, 2))

                      # draw = ImageDraw.Draw(img)
                      # draw.polygon(polygon, outline=(0, 255, 0), width=5)
                      cv2.polylines(img_s, [pts_s], True, color=(0, 0, 255), thickness=2)  # draw polygon

                      # -- Find Calculate --------------------
                      area_px_s_dummy.append(int(cv2.contourArea(polygon_s)))

                      i = i + 1

                  # --To find bounding box--------
                  #print(results_s)

                  # ------SIDE-DETECT----------------------------------------------------
                  for r in results_s:
                      boxes = r.boxes
                      i = 0
                      for box in boxes:
                          # Bounding Box
                          x1, y1, x2, y2 = box.xyxy[0]
                          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                          w, h = x2 - x1, y2 - y1
                          cx, cy = x1 + w // 2, y1 + h // 2

                          #cv2.rectangle(img_s, (x1, y1), (x2, y2), (0, 0, 255), 2)
                          # Confidence
                          conf = math.ceil((box.conf[0] * 100)) / 100
                          # Class Name
                          cls = int(box.cls[0])

                          currentClass_s = classNames[cls]
                          #print("SIDE_Curret class",currentClass_s)

                          if currentClass_s != "person" and conf > 0.3 and 650 > cy:
                              cvzone.putTextRect(img_s, f'{currentClass_s} {conf} Area:{format(area_px_s_dummy[i])}', (max(0, x1), max(35, y1)),
                                                     scale=1, thickness=1)  # Class Name
                              allArray_s.append([x1, y1, x2, y2, currentClass_s])
                              currentArray_s = np.array([x1, y1, x2, y2, conf])
                              detections_s = np.vstack((detections_s, currentArray_s))
                              area_px_s.append(area_px_s_dummy[i])
                              found_ele_s.append([str(currentClass_s),cx,cy])
                          i = i+1
              else:
                  break

              # ---TOP VIEW-MASK-----------------------------------------------------

              if success_t:
                  # Detect movement-----------------
                  results_m = model(forground)

                  for r in results_m:
                      boxes = r.boxes
                      for box in boxes:
                          # Bounding Box
                          x1, y1, x2, y2 = box.xyxy[0]
                          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                          w, h = x2 - x1, y2 - y1
                          cx, cy = x1 + w // 2, y1 + h // 2
                          cv2.circle(forground, (cx, cy), 7, (0, 0, 255), cv2.FILLED)

                          # frameCopy = apply_mask(frameCopy, (x1+50, y1+50), (x2-50, y2-50))
                          # frameCopy = apply_mask(frameCopy, (cx - 50, cy - 50), (cx + 50, cy + 50))
                          # cv2.rectangle(img_s, (x1, y1), (x2, y2), (0, 0, 255), 2)
                          # Confidence
                          conf = math.ceil((box.conf[0] * 100)) / 100
                          # Class Name
                          cls = int(box.cls[0])
                          currentClass = classNames[cls]
                          # print("<-----POINTS---->",currentClass, cx, cy)

                          if currentClass != "person" and conf > 0.3:
                              cvzone.putTextRect(forground, f'{currentClass} {conf} ',
                                                 (max(0, x1), max(35, y1)),
                                                 scale=1, thickness=1)  # Class Name
                              movement_detector.append([currentClass, cx, cy])

                  # -----------------------------------------------------------

                  # Run Yolov8 inference on the frame
                  results_t = model(img_t)
                  area_px_t = []
                  area_px_t_dummy = []
                  found_ele_t = []

                  # visulaize the results on the frame without bounding box
                  annotated_frame = results_t[0].plot(boxes=False)

                  # --- find mask -----------------------------------------
                  l = 0
                  masks = results_t[0].masks
                  if results_t[0].masks is not None:
                      l = len(masks)
                  print(l)
                  i = 0
                  while i < l:
                      mask1 = masks[i]
                      mask = mask1.cpu().data[0].numpy()
                      polygon = mask1.xy[0]
                      shape = mask1.shape

                      mask_img = Image.fromarray(mask, "I")
                      pts = np.array([polygon], np.int32)
                      pts = pts.reshape((-1, 1, 2))

                      # draw = ImageDraw.Draw(img)
                      # draw.polygon(polygon, outline=(0, 255, 0), width=5)
                      cv2.polylines(img_t, [pts], True, color=(0, 0, 255), thickness=2)  # draw polygon

                      # -- Find Calculate --------------------
                      area_px_t_dummy.append(int(cv2.contourArea(polygon)))

                      i = i + 1

                  # --To find bounding box--------
                  # print(results)

                  # print("-----movement----",movement_detector)
                  # print("-----area_px_t_dummy----",area_px_t_dummy)
                  # ------TOP-DETECT----------------------------------------------------
                  for r in results_t:
                      boxes = r.boxes
                      # print("-----len--------",len(boxes))
                      i = 0
                      for box in boxes:
                          # Bounding Box
                          x1, y1, x2, y2 = box.xyxy[0]
                          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                          w, h = x2 - x1, y2 - y1
                          w, h = x2 - x1, y2 - y1
                          cx, cy = x1 + w // 2, y1 + h // 2
                          cv2.rectangle(img_t, (x1, y1), (x2, y2), (0, 0, 255), 2)

                          cv2.circle(img_t, (cx, cy), 7, (0, 255, 0), cv2.FILLED)

                          # Confidence
                          conf = math.ceil((box.conf[0] * 100)) / 100
                          # Class Name
                          cls = int(box.cls[0])
                          currentClass_t = classNames[cls]
                          # print("------TOP--cx,cy------------",cx,cy)
                          # print("TOP_Curret class", currentClass_t)
                          for mv in movement_detector:
                              if currentClass_t == mv[0] and conf > 0.3 and cx - 100 < mv[1] < cx + 100 and cy - 100 < mv[
                                  2] < cy + 100:
                                  cvzone.putTextRect(img_t, f'{currentClass_t} {conf} Area:{format(area_px_t_dummy[i])}',
                                                     (max(0, x1), max(35, y1)),
                                                     scale=1, thickness=1)  # Class Name
                                  allArray_t.append([x1, y1, x2, y2, currentClass_t])
                                  currentArray_t = np.array([x1, y1, x2, y2, conf])
                                  detections_t = np.vstack((detections_t, currentArray_t))
                                  area_px_t.append(area_px_t_dummy[i])
                                  found_ele_t.append([str(currentClass_t), cx, cy])

                          i = i + 1
                          # print("----area_px_top_____",area_px_t)
              else:
                  break


###################################################3
            
######################################################

              resultsTracker_s = tracker_s.update(detections_s)
              resultsTracker_t = tracker_t.update(detections_t)

              cnt_s = 0
              cnt_t = 0


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

                            # ---------Overlap Bounding Box---------
                            ovr_flg_t = 0
                            for ovr_ele in allArray_t:
                                lx1 = ovr_ele[0]
                                ly1 = ovr_ele[1]
                                lx2 = ovr_ele[2]
                                ly2 = ovr_ele[3]
                                ovr_cls = ovr_ele[4]
                                if (
                                        x1 >= lx1 and y1 >= ly1 and x2 <= lx2 and y2 <= ly2 and ovr_cls != currentClass_t):
                                    #    if (ovr_cls=="Maa_Juice"):
                                    ovr_flg_t = 1
                                    #    break
                                    #        else:
                                    #    Products_added_t.append(currentClass)
                                    break

                            if ovr_flg_t == 0:
                                Products_removed_t.append(currentClass_t)

                            print("remmove")

                            Result = []
                            # -----------------Union Products----------------
                            temp_top = []
                            temp_side = []
                            union_pro = []
                            temp_top.extend(Products_removed_s)
                            temp_side.extend(Products_removed_t)
                            union_pro.extend(temp_top)
                            for element in temp_side:
                                if element in temp_top:
                                    temp_top.remove(element)
                                else:
                                    union_pro.append(element)
                            print(union_pro)
                            # union_pro = list(set(Products_removed_t + Products_removed_s))

                            print(Products_removed_t, Products_removed_s, Hide_remove)
                            print(Final, Free)

                            # ------------Repeated Deletion ----------------
                            if currentClass_t in Products_added_t:
                                Products_added_t.remove(currentClass_t)

                            else:

                                # ----------------DISCOUNT OFFER-ALONE-----------------------------
                                if Original_Product in union_pro and Discount_Product in union_pro and len(union_pro) == 2:
                                    Final.remove(Original_Product)
                                    Free.remove(Discount_Product)
                                    Hide_remove.extend(Products_removed_s)
                                    Hide_remove_time = int(fps)
                                    Result.extend(union_pro)
                                # -----------------------------------------------
                                else:
                                    if len(Products_removed_s) == 0:

                                        if currentClass_t in Hide_remove:
                                            Hide_remove.remove(currentClass_t)
                                        else:
                                            for ele in Products_removed_t:
                                                if ele in Final:
                                                    Final.remove(ele)
                                                    Result.append(ele)
                                                else:
                                                    Free.remove(ele)
                                                    Result.append(ele)

                                    else:
                                        intersection = list(set(Products_removed_s) & set(Products_removed_t))
                                        print(intersection)
                                        if len(intersection) == 0:

                                            if currentClass_t in Hide_remove:
                                                Hide_remove.remove(currentClass_t)
                                            else:
                                                Hide_remove.extend(Products_removed_s)
                                                Hide_remove_time = int(fps)
                                                for ele in union_pro:
                                                    if ele in Final:
                                                        Final.remove(ele)
                                                        Result.append(ele)
                                                    else:
                                                        Free.remove(ele)
                                                        Result.append(ele)


                                        else:
                                            if len(Products_removed_s) >= len(Products_removed_t):

                                                Result.extend(union_pro)

                                                for ele in Result:
                                                    if ele in Final:
                                                        Final.remove(ele)
                                                    else:
                                                        Free.remove(ele)

                                                for H_ele in Products_removed_t:
                                                    if H_ele in Products_removed_s:
                                                        Products_removed_s.remove(H_ele)
                                                Hide_remove.extend(Products_removed_s)
                                                Hide_remove_time = int(fps)
                            Products_removed_t = []
                            Products_removed_s = []

                            # ----------------DISCOUNT-OFFER + OTHER---------------
                            Free_tmp = []
                            print(Result)
                            Org_Cnt = 0
                            for ele in Result:
                                print(ele, Original_Product)
                                if ele == Original_Product:
                                    Org_Cnt = Org_Cnt + 1

                            print(Org_Cnt)

                            for ele in Result:
                                if ele == Discount_Product and Org_Cnt > 0:
                                    Org_Cnt = Org_Cnt - 1
                                    Free_tmp.append(Discount_Product)
                            print(Free)

                            #--UPDATE DATABASE----------
                            for ele in Result:
                                value_to_select = ele

                                for r in Segment_remove:
                                    if value_to_select in r:
                                        value_to_select = r

                                    product = value_to_select
                                    doc_ref=db.collection(tb).document(product)
                                    doc=doc_ref.get()

                                    if doc.exists:
                                            doc_ref.update({'Stock':firestore.Increment(-1)})
                                            doc = doc_ref.get()
                                            doc_data = doc.to_dict()
                                            stock = doc_data.get('Stock')
                                            amt = doc_data.get('Amount')
                                            #det = doc_data.get('Category')
                                            det = product
                                            kpi4_text.write(f"<h1  style='color:red;'>{det}</h1>",unsafe_allow_html=True)
                                            kpi3_text.write(f"<h1  style='color:red;'>{stock}</h1>",unsafe_allow_html=True)
                                            kpi2_text.write(f"<h1 style='color:red;'>{amt}</h1>",unsafe_allow_html=True)

                                            if len(Free_tmp)>0 and ele==Discount_Product:
                                                Free_tmp.remove(Discount_Product)
                                            
                                            else:
                                                current_total = current_total - row[1]
                                                U_Final.remove(value_to_select)
                                            
                                            kpi1_text.write(f"<h1 style='color:white;'>{'{:.1f}'.format(current_total)}</h1>",unsafe_allow_html=True)
                                    else:
                                            st.title("NOT FOUND")



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
                

              
              for result in resultsTracker_s:
              
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    for r in allArray_s:
                        if (r[0] - 50 < x1 < r[0] + 50 and r[1] - 50 < y1 < r[1] + 50 and r[2] - 50 < x2 < r[2] + 50 and r[3] - 50 < y2 < r[3] + 50):
                            currentClass_s = r[4]

                    w, h = x2 - x1, y2 - y1
                    
                    cx, cy = x1 + w // 2, y1 + h // 2
                    cv2.circle(img_s, (cx, cy), 7, (0, 0, 255), cv2.FILLED)
          
                    #-----BOUNDING BOX FOR OBJECTS INSIDE CART--------------------------
                    if top_limits1_s[1] < cy:
                        cvzone.putTextRect(img_s, f' {int(id)}', (max(0, cx), max(35, cy)), scale=1.5, thickness=2,offset=10)
                        cv2.rectangle(img_s, (x1, y1), (x2, y2), (0, 255, 0), 2)


                    # TOP-OUTER LIMIT-----------------------------------------------------------

                    if top_limits1_s[0] < cx < top_limits1_s[2] and top_limits1_s[1] - 25 < cy < top_limits1_s[3] + 25:
                        cv2.line(img_s, (top_limits1_s[0], top_limits1_s[1]), (top_limits1_s[2], top_limits1_s[3]),(0, 255, 0), 5)

                        if out_line_s.count(id) == 0 and in_line_s.count(id) == 0:
                            # if totalCount.count(id) == 0:
                            out_line_s.append(id)
                            print("out-1")
                        else:
                            # REMOVE --------------------------------
                            if out_line_s.count(id) == 0 and in_line_s.count(id) == 1:
                                print("out-2")
                                Total_products_s = Total_products_s - 1
                                in_line_s.remove(id)


                    # TOP-INNER-LIMIT----------------------------------------------------------------

                    if top_limits2_s[0] < cx < top_limits2_s[2] and top_limits2_s[1] - 25 < cy < top_limits2_s[3] + 25:
                        cv2.line(img_s, (top_limits2_s[0], top_limits2_s[1]), (top_limits2_s[2], top_limits2_s[3]),(0, 255, 0), 5)

                        if in_line_s.count(id) == 0 and out_line_s.count(id) == 0:
                            # if totalCount.count(id) == 0:
                            print("in-1")
                            in_line_s.append(id)

                        else:
                            # ADD --------------------------------
                            if in_line_s.count(id) == 0 and out_line_s.count(id) == 1:
                                print("in-2")
                                Total_products_s = Total_products_s + 1
                                out_line_s.remove(id)

              
              cv2.putText(img_s, "TOTAL: " + str(int(current_total)), (20, 360), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 5)
              cv2.putText(img_t, "TOTAL: " + str(int(current_total)), (20, 360), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 5)

              cv2.putText(img_s, "Count: " + str(len(U_Final)), (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 5)
              cv2.putText(img_s, "Free: " + str(len(Free)), (20, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 5)

              cv2.putText(img_t, "Count: " + str(len(U_Final)), (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 5)
              cv2.putText(img_t, "Free: " + str(len(Free)), (20, 120), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 5)      

              #-------BOUNDARY LINES--------------------------

              cv2.line(img_t, (left_limits1[0], left_limits1[1]), (left_limits1[2], left_limits1[3]), (0, 0, 255), 3)
              cv2.line(img_t, (left_limits2[0], left_limits2[1]), (left_limits2[2], left_limits2[3]), (255, 0, 0), 3)
    
              cv2.line(img_t, (right_limits1[0], right_limits1[1]), (right_limits1[2], right_limits1[3]), (0, 0, 255), 3)
              cv2.line(img_t, (right_limits2[0], right_limits2[1]), (right_limits2[2], right_limits2[3]), (255, 0, 0), 3)
    
              cv2.line(img_t, (top_limits1[0], top_limits1[1]), (top_limits1[2], top_limits1[3]), (0, 0, 255), 3)
              cv2.line(img_t, (top_limits2[0], top_limits2[1]), (top_limits2[2], top_limits2[3]), (255, 0, 0), 3)
    
              cv2.line(img_t, (bottom_limits1[0], bottom_limits1[1]), (bottom_limits1[2], bottom_limits1[3]), (0, 0, 255), 3)
              cv2.line(img_t, (bottom_limits2[0], bottom_limits2[1]), (bottom_limits2[2], bottom_limits2[3]), (255, 0, 0), 3)

              cv2.line(img_s, (top_limits1_s[0], top_limits1_s[1]), (top_limits1_s[2], top_limits1_s[3]), (0, 0, 255), 3)
              cv2.line(img_s, (top_limits2_s[0], top_limits2_s[1]), (top_limits2_s[2], top_limits2_s[3]), (255, 0, 0), 3)
              cv2.line(img_s, (top_limits3_s[0], top_limits3_s[1]), (top_limits3_s[2], top_limits3_s[3]), (255, 0, 0), 3)
                  

              stframe_t.image(img_t, channels='BGR', use_column_width=True)
              stframe_s.image(img_s, channels='BGR', use_column_width=True)

    
    st.title("!! DONE !!")

