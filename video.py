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
# import cv2
# from PIL import Image
# import tempfile
# import os

# def main():
#     st.title("Streamlit Video Player")

#     uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

#     if uploaded_file is not None:
#         # Save the uploaded file to a temporary location
#         temp_file = tempfile.NamedTemporaryFile(delete=False)
#         temp_file.write(uploaded_file.read())

#         # Display the video frame by frame
#         stop_button = st.button("Stop")
#         stframe = st.empty()
#         cap = cv2.VideoCapture(temp_file.name)

#         while cap.isOpened() and not stop_button:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             # Convert the frame from BGR to RGB
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             # Display the frame
#             #st.image(Image.fromarray(rgb_frame))
#             stframe.image(Image.fromarray(rgb_frame))

#         # Close the video capture object
#         cap.release()

#         # Remove the temporary file
#         os.unlink(temp_file.name)

# if __name__ == "__main__":
#     main()
# STORAGE --> webrtc ---------------------------------------
import streamlit as st
import os

def main():
    st.title("Streamlit Video Player")

    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        # Display the video using st.video
        st.video("temp_video.mp4")

        # Remove the temporary file after displaying the video
        os.remove("temp_video.mp4")

if __name__ == "__main__":
    main()
