# pip install ultralytics --upgrade
from ultralytics import YOLO
yaml_file_path='data.yaml'
yolo_model_name="yolov8s.pt"
model = YOLO(yolo_model_name)
epochs=100
model.train(data=yaml_file_path, epochs=epochs)
model.save('botteldetection_v7n.pt')
