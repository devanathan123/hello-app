#VIDEO-------------------------------------------
# import streamlit as st
# from ultralytics import YOLO
# import cv2
# def main():
#     st.title("Streamlit Video Player")
#
#     # Google Drive video URL
#     video_url = "https://firebasestorage.googleapis.com/v0/b/fir-ec695.appspot.com/o/t-13%2Finput_side.mp4?alt=media&token=0cf29a63-b98b-451a-a9f5-7e2d71caf9c3"  # Replace "your-file-id" with the actual file ID
#
#     # Display the video
#     st.video(video_url)
#
# if __name__ == "__main__":
#     main()
#VIDEO CAPTURE------------------------------------------------
# import streamlit as st
# import cv2
# import numpy as np

# st.title("Streamlit Webcam Player")
    
# def main():
#     video_url = "https://firebasestorage.googleapis.com/v0/b/fir-ec695.appspot.com/o/t-13%2Finput_side.mp4?alt=media&token=0cf29a63-b98b-451a-a9f5-7e2d71caf9c3"  # Replace "your-file-id" with the actual file ID
#     cap = cv2.VideoCapture(video_url)
#     load = st.button("STOP")
#     stframe = st.empty()

#     left_limits1 = [0, 450, 1920,450]
#     left_limits2 = [0, 500, 1920,500]

#     while not load:
#         success, img = cap.read()

#         cv2.line(img, (left_limits1[0], left_limits1[1]), (left_limits1[2], left_limits1[3]), (0, 0, 255), 5)
#         cv2.line(img, (left_limits2[0], left_limits2[1]), (left_limits2[2], left_limits2[3]), (255, 0, 0), 5)


#         if not success:
#             break
 
#         # Display the frame in the video element
#         stframe.image(img, channels='BGR', use_column_width=True)

    
#     st.title("DONE!!")

# if __name__ == "__main__":
#     main()
#CAM PERMISSION---------------------------------------
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer

# def main():
#     st.title("Webcam Video Stream")

#     webrtc_ctx = webrtc_streamer(
#         key="example",
#         video_transformer_factory=None,  # No processing needed
#         async_transform=False,  # No need for async processing
#     )

#     if webrtc_ctx.state.playing:
#         st.write("Streaming webcam feed...")

# if __name__ == "__main__":
#     main()
#Storage file --CV2 ----------------------------------------------
# import streamlit as st
# import os
# import cv2
# def main():
#     st.title("Streamlit Video Player")

#     uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

#     if uploaded_file is not None:
#         # Save the uploaded file to a temporary location
#         with open("temp_video.mp4", "wb") as temp_file:
#             temp_file.write(uploaded_file.read())

#         # Display the video frame by frame
#         stop_button = st.button("Stop")
#         stframe = st.empty()
#         cap = cv2.VideoCapture(temp_file.name)

#         while cap.isOpened() and not stop_button:
#             success, img= cap.read()
#             if not success:
#                 break

#             stframe.image(img, channels='BGR', use_column_width=True)


#         st.title("DONE !!")

# if __name__ == "__main__":
#     main()
#NEW ML MODEL ---------------------------------------------------
import streamlit as st
import os
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import math
from pathlib import Path

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())


def main():
    
    # ML Model config
    MODEL_DIR = ROOT / 'YOLO-Weights'
    DETECTION_MODEL = MODEL_DIR / 'seg3n_25.pt'
    model = YOLO(DETECTION_MODEL)

    classNames = ['Cinthol_Soap', 'Hamam_Soap', 'Him_Face_Wash', 'Maa_Juice', 'Mango', 'Mysore_Sandal_Soap',
                  'Patanjali_Dant_Kanti', 'Tide_Bar_Soap', 'ujala_liquid']
    st.title("Streamlit Video Player")

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp_video.mp4", "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Display the video frame by frame
        stop_button = st.button("Stop")
        stframe = st.empty()
        cap = cv2.VideoCapture(temp_file.name)

        while cap.isOpened() and not stop_button:
            success, img= cap.read()
            results = model(img, stream=True)
            detections = np.empty((0,5))
            allArray = []
            currentClass = ""

            if success:

                for r in results:
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

                        currentClass = classNames[cls]

                        if currentClass != "person" and conf > 0.3 and 650 > cy:

                            cvzone.putTextRect(img, f'{currentClass} {conf}',
                                               (max(0, x1), max(35, y1)),
                                               scale=3, thickness=3)  # Class Name
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                stframe.image(img, channels='BGR', use_column_width=True)

            else:
                break


        st.title("DONE !!")

if __name__ == "__main__":
    main()
