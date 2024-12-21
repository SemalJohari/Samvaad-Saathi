import streamlit as st
import cv2
import torch
from models.common import DetectMultiBackend  # YOLOv5 detection module
from utils.plots import Annotator, colors
from utils.general import non_max_suppression
import numpy as np
import pathlib
import os

# Adjust pathlib for compatibility
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Samvaad Saathi: Your Sign Language Translator")

# Introduction
introduction = """
### About This Project:

This is a machine learning app for real-time Sign Language Detection and Translation,
built using the YOLOv5 model and implemented using PyTorch and OpenCV libraries.
It can translate common phrases in sign language such as 'hello', 'thank you', etc.
Click on the 'Start' button to get started!
"""
st.markdown(introduction)

# Sidebar for phrases
with st.sidebar:
    static_folder = "static"
    with st.expander("## 💡 Some Common phrases", expanded=False):
        phrases = [
            "hello", "thank you", "i love you", "please", "sorry", "fine", "yes",
            "no", "repeat", "eat", "help", "what"
        ]

        cols = st.columns(2)  # Create a 2-column grid

        for idx, phrase in enumerate(phrases):
            img_path = os.path.join(static_folder, f"{phrase}.jpg")
            col = cols[idx % 2]  # Alternate between columns

            if os.path.exists(img_path):
                with col:
                    st.image(img_path, caption=f"{phrase.capitalize()}", use_container_width=True)
            else:
                with col:
                    st.warning(f"Image for '{phrase}' not found.")

# Model configuration
weights_path = "best.pt"
conf_thresh = 0.5
img_size = 640

# Cache the model
@st.cache_resource
def load_model(weights_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectMultiBackend(weights_path, device=device)
    return model

model = load_model(weights_path)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape."""
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clip(min=0, max=max(img0_shape))
    return coords

# Initialize session state for video capturing
if "capturing" not in st.session_state:
    st.session_state["capturing"] = False

# Start and Stop buttons
col1, col2 = st.columns(2)
if col1.button("Start Video Capture"):
    st.session_state["capturing"] = True
if col2.button("Stop"):
    st.session_state["capturing"] = False

# Video Capture and Display
video_feed = st.empty()

if st.session_state["capturing"]:
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and st.session_state["capturing"]:
        ret, frame = cap.read()

        if not ret:
            st.error("Failed to capture video.")
            break

        # Preprocess frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (img_size, img_size))
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Run inference
        with torch.no_grad():
            pred = model(img_tensor)
            pred = non_max_suppression(pred, conf_thresh, iou_thres=0.45)

        # Annotate the frame
        annotator = Annotator(frame, line_width=2, example=str(model.names))

        if pred[0] is not None and len(pred[0]):
            for det in pred[0]:
                if det is not None and len(det):
                    if det.dim() == 1:
                        det = det.unsqueeze(0)

                    det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], frame.shape).round()

                    for *xyxy, conf, cls in reversed(det):
                        label = f"{model.names[int(cls)]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(int(cls), True))

        # Display annotated frame
        annotated_frame = annotator.result()
        video_feed.image(annotated_frame, channels="BGR")

    cap.release()
    st.success("Video capture stopped.")
else:
    st.info("Press 'Start Video Capture' to begin.")
