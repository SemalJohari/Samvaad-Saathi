# Samvaad Saathi: Your Sign Language Translator

This is a machine learning app for real-time Sign Language Detection and Translation, 
built using the YOLOv5 model and implemented using PyTorch and OpenCV libraries. It 
can translate common phrases in sign language such as 'hello', 'sorry', 'thank you', etc. 

_The static web application can be accessed at [Streamlit App](https://samvaadsaathi.streamlit.app/)
since Streamlit Cloud doesn't support webcam. For dynamic access, visit the [Installation](#Installation),
[Local-Usage](#Local-Usage) and [App-Usage](#App-Usage) sections._

## Table of Contents

- [Features](#Features)
- [Tech-Stack](#Tech-Stack)
- [Screenshots](#Screenshots)
- [Installation](#Installation)
- [Local-Usage](#Local-Usage)
- [App-Usage](#App-Usage)
- [Tip](#Tip)

## Features

1. Real-time sign language detection and translation
2. Buttons for starting and stopping video capture
3. Common Sign Language phrases dictionary in the side bar

## Tech-Stack

1. Streamlit Framework for Frontend
2. YOLOv5 as the base model for sign language detection
3. Python modules:
   1. Pytorch for implementation of YOLOv5 model
   2. OpenCV for video capturing and data collection
   3. Weights and Biases (WandB) for model training
   4. OS for executing commands on the local machine
   5. And many more packages like sys, uuid, time, pathlib, argparse, etc.
4. VSCode and Google Colab for development requirements
5. Webcam and Windows OS for hardware requirements

## Screenshots

<p align="center">
   <img src="https://github.com/user-attachments/assets/5f89fbca-0208-4666-b665-b84e60e73497" alt="Screenshot (19)" width="300"/>
   <img src="https://github.com/user-attachments/assets/d72f1cbb-5f5a-4947-9420-18cab149e962" alt="Screenshot (194)" width="300"/>
    <img src="https://github.com/user-attachments/assets/9f59d34a-8193-4307-9021-1b938b1e6825" alt="Screenshot (195)" width="300"/>
   <img src="https://github.com/user-attachments/assets/f47309b3-96ec-425f-a073-e6588969526e" alt="Screenshot (196)" width="300"/>
   <img src="https://github.com/user-attachments/assets/08f02f55-0848-4677-8264-286002dc95db" alt="Screenshot (197)" width="300"/>
</p>

## Installation

Follow these steps to clone and set up the repository locally:

1. Clone the repository:

   ```bash
   git clone https://github.com/SemalJohari/Samvaad-Saathi.git

2. Navigate to the project directory:

   ```bash
   cd Samvaad-Saathi

4. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate

6. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

## Local-Usage

To run the project locally, start the development server or application using:
    
    python run.py

## App-Usage

To run the project at using a web application with a local URL, start the 
development server or application using:
    
    streamlit run app.py

## Tip

For running the project on Windows, add the following lines of code immediately 
after importing libraries in app.py and detect.py to switch from the Posix Path 
(for MacOS) to Windows Path (for Windows OS) using the pathlib library:

   ```python
   temp = pathlib.PosixPath
   pathlib.PosixPath = pathlib.WindowsPath
