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
#WEBCAM------------------------------------------------
# import streamlit as st
# import cv2
# import numpy as np

# st.title("Streamlit Webcam Player")
    
# def main():
#     video_url = "https://firebasestorage.googleapis.com/v0/b/fir-ec695.appspot.com/o/t-13%2Finput_side.mp4?alt=media&token=0cf29a63-b98b-451a-a9f5-7e2d71caf9c3"  # Replace "your-file-id" with the actual file ID
#     cap = cv2.VideoCapture(video_url)
#     load = st.button("STOP")
#     stframe = st.empty()

#     while not load:
#         success, img = cap.read()
#         if not success:
#             break

#         # Display the frame in the video element
#         stframe.image(img, channels='BGR', use_column_width=True)

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
# import cv2
# import tempfile

# st.title("Streamlit Webcam Player")

# def main():
#     uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

#     if uploaded_file is not None:
#         # Save the uploaded file to a temporary location
#         with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#             temp_file.write(uploaded_file.read())
#             temp_file_path = temp_file.name

#         # Open the video file using OpenCV
#         cap = cv2.VideoCapture(temp_file_path)
#         load = st.button("STOP")
#         stframe = st.empty()

#         while not load:
#             success, img = cap.read()
#             if not success:
#                 break

#             # Display the frame in the video element
#             stframe.image(img, channels='BGR', use_column_width=True)

#         # Release the video capture object and delete the temporary file
#         cap.release()
#         del temp_file

# if __name__ == "__main__":
#     main()
# STORAGE --> webrtc ---------------------------------------
import streamlit as st
from streamlit_webrtc import webrtc_streamer

def main():
    st.title("Streamlit Webcam Player")

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

    if uploaded_file is not None:
        video_file = uploaded_file.name
        webrtc_ctx = webrtc_streamer(
            key="example",
            video_processor_factory=None,  # No need for custom processing
            mode="file",
            file_handler=uploaded_file,
        )

        if webrtc_ctx.video_processor:
            st.write("Streaming video...")
        else:
            st.warning("Error loading video.")

if __name__ == "__main__":
    main()
