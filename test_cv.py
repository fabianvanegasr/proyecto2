import cv2
import torch
from ultralytics import YOLO

print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)

model = YOLO("yolov8n.pt")
result = model.predict( source='https://github.com/fabianvanegasr/proyecto2/blob/main/personas.jpg',
                       save=True)

print("Parece que funcionó")

print("YOLO listo")
