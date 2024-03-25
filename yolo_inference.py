import torch
import torchvision

from ultralytics import YOLO

model = YOLO('models/best.pt')

result = model.predict('inputs/Demo_2.mp4', conf=0.2, save=True)

# Visualize the predictions of the model on various objects within given inputs
print(result)
print("Boxes: ")
for box in result[0].boxes:
  print(box)