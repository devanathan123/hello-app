#VIDEO-------------------------------------------
# import streamlit as st
# from ultralytics import YOLO
# import cv2
# def main():
#     st.title("Streamlit Video Player")
#
#     # Google Drive video URL
#     #video_url = "https://firebasestorage.googleapis.com/v0/b/fir-ec695.appspot.com/o/t-13%2Finput_side.mp4?alt=media&token=0cf29a63-b98b-451a-a9f5-7e2d71caf9c3"  # Replace "your-file-id" with the actual file ID
#
#     # Display the video
#     #st.video(video_url)
#
#     cap = cv2.VideoCapture(0)
#     while True:
#         success, img = cap.read()
#         cv2.imshow("image",img)
#         cv2.waitKey(1)
#
# if __name__ == "__main__":
#     main()
#WEBCAM------------------------------------------------
import streamlit as st
import cv2
import numpy as np

st.title("Streamlit Webcam Player")
    
def main():
    cap = cv2.VideoCapture(0)
    load = st.button("STOP")
    stframe = st.empty()

    while not load:
        success, img = cap.read()
        if not success:
            break

        # Display the frame in the video element
        stframe.image(img, channels='BGR', use_column_width=True)

if __name__ == "__main__":
    main()
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
