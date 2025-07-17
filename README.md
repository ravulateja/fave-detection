# fave-detection

🎯 Real-Time Face Detection using MTCNN and OpenCV
This project demonstrates how to perform real-time face detection using a webcam feed with the MTCNN (Multi-task Cascaded Convolutional Networks) model from the facenet-pytorch library. The implementation uses OpenCV, PIL, and PyTorch to capture frames, detect faces, and display the results in real-time.

🧰 Requirements
Install the required packages with:


pip install facenet-pytorch opencv-python pillow numpy tensorflow streamlit flask scikit-learn

Make sure your environment also has PyTorch installed. You can install it from pytorch.org.

🚀 How to Run
Run the Python script directly:


python face_detection.py
Press q to quit the webcam feed.

📂 Code Overview

# Import dependencies
import cv2
from PIL import Image
from facenet_pytorch import MTCNN

# Load MTCNN model
mt = MTCNN(keep_all=True)

# Start webcam
cap = cv2.VideoCapture(0)


🧠 Model Details
MTCNN is a deep learning-based face detection model that uses a cascaded architecture of three stages of convolutional networks.

Provided by the facenet-pytorch library, it is optimized for performance and can detect multiple faces in a single frame.

✅ Features
Real-time face detection via webcam

Uses GPU if available (via PyTorch)

Multi-face detection (keep_all=True)

Simple and efficient code structure

📌 TODO
Add support for face alignment and cropping

Integrate Streamlit or Flask for web-based UI (placeholder imports already in code)

Save cropped faces to disk

Add face recognition support

📄 License
This project is open-source under the MIT License.
