import cv2
import torch
from ultralytics import YOLO

print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)

model = YOLO("yolov8n.pt")
#result = model.predict( source='https://ultralytics.com/images/bus.jpg')
result = model.predict( source='https://es.vecteezy.com/foto/35712186-ai-generado-un-grupo-de-domestico-mascota-perros-y-gatos-colgando-patas-terminado-repisa')
print("Parece que funcionó")

print("YOLO listo")
