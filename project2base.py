import sklearn as sl
from PIL import Image,ImageDraw
import cv2
import flask
import tensorflow as tf 
import numpy as np
import streamlit as st 
from facenet_pytorch import MTCNN
mt=MTCNN(keep_all=True)
cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if not ret:
        break

    img = Image.fromarray(frame)


    box,probs=mt.detect(img)
    if box is not None:
        for b in box:
            x1, y1, x2, y2 = [int(coord) for coord in b]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("video",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()