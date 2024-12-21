import streamlit as st
import cv2
import torch
from models.common import DetectMultiBackend  # YOLOv5 detection module
from utils.plots import Annotator, colors
from utils.general import non_max_suppression
import numpy as np
import pathlib
import os

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Samvaad Saathi: Your Sign Language Translator")

introduction = """
### About This Project:

This is a machine learning app for real time Sign Language Detection and Translation,
built using YOLOv5 model and implemented using Pytorch and OpenCV libraries.
It can translate common phrases in sign language such as 'hello', 'thank you', etc.
Click on the 'Start' button to get started!
"""
st.markdown(introduction)

with st.sidebar:
    static_folder = "static"
    with st.expander("## 💡 Some Common phrases", expanded=False):
        phrases = [
            "hello", "thank you", "i love you", "please", "sorry", "fine", "yes",
            "no", "repeat", "eat", "help", "what"
        ]
        
        cols = st.columns(2)  # Create 2 columns for the grid

        for idx, phrase in enumerate(phrases):
            img_path = os.path.join(static_folder, f"{phrase}.jpg")
            
            # Determine the column for the current image
            col = cols[idx % 2]  # Alternate between columns

            # Display the image and its caption
            if os.path.exists(img_path):
                with col:
                    st.image(img_path, caption=f"{phrase.capitalize()}", use_container_width=True)
            else:
                with col:
                    st.warning(f"Image for '{phrase}' not found.")

# Sidebar for Model Settings
weights_path = "best.pt"
conf_thresh = 0.5
img_size = 640

# Initialize the YOLOv5 model
@st.cache_resource
def load_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(weights_path, device=device)
    return model

model = load_model(weights_path)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape."""
    if ratio_pad is None:  # Calculate from img1_shape and img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clip(min=0, max=max(img0_shape))  # clip coords
    return coords

# Sidebar button to start the video capture
start_button_pressed = st.button("Start Video Capture")

# If the Start button is pressed, initiate the video capture
if start_button_pressed:
    # Start Video Feed
    video_feed = st.empty()
    stop_button_pressed = st.button("Stop")

    # Video Capture
    cap = cv2.VideoCapture(0)  # Use your laptop camera
    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()

        if not ret:
            st.error("Failed to capture video.")
            break

        # Preprocess frame for YOLOv5
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (img_size, img_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Run inference
        with torch.no_grad():
            pred = model(img_tensor)
            pred = non_max_suppression(pred, conf_thresh, iou_thres=0.45)

        # Initialize Annotator for drawing bounding boxes
        annotator = Annotator(frame, line_width=2, example=str(model.names))

        if pred[0] is not None and len(pred[0]):
            for det in pred[0]:
                if det is not None and len(det):
                    if det.dim() == 1:  # Handle 1D tensor case
                        det = det.unsqueeze(0)  # Convert to 2D tensor

                    # Rescale the bounding box coordinates
                    det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        label = f"{model.names[int(cls)]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))

        # Display annotated frame
        annotated_frame = annotator.result()
        video_feed.image(annotated_frame, channels="BGR")

        # Exit if "Stop" button is pressed
        if stop_button_pressed:
            break

    cap.release()
    st.success("Detection stopped.")