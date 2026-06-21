from ultralytics import YOLO
 
model = YOLO('model/best.pt')
model.export(format="onnx")