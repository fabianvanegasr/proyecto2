import cv2
import torch
from ultralytics import YOLO

print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)

model = YOLO("yolov8n.pt")
#result = model.predict( source='https://ultralytics.com/images/bus.jpg')
result = model.predict( source='https://github.com/fabianvanegasr/proyecto2/blob/main/personas.jpg')
print("Parece que funcionó")

print("YOLO listo")

# Obtener la imagen procesada
# `result[0].plot()` genera una imagen con las predicciones (cuadros delimitadores, etiquetas, etc.)
# Puedes guardar esa imagen en una carpeta.

output_image = result[0].plot()  # Esto genera la imagen procesada con las predicciones

# Guardar la imagen procesada en la carpeta 'predicciones'
output_path = 'runs/detect/predict/personas_predicciones.jpg'
cv2.imwrite(output_path, output_image)

print("Imagen con predicciones guardada en:", output_path)