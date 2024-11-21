import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
import numpy as np

# load model    
device = select_device('cpu')  # GPU
model = attempt_load('yolov7.pt', map_location=device)

# read video file
cap = cv2.VideoCapture('data/test1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

# frame processing before detection
    img = cv2.resize(frame, (640, 640))  # change size yolo model input size
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    img = np.ascontiguousarray(img)  # sort memory
    img = torch.from_numpy(img).to(device).float() / 255.0  # convert to tensor and normalize
    img = img.unsqueeze(0)  # add batch resource (batch_size, channels, height, width)

    # detect object yolo model
    with torch.no_grad():  # detect object without gradient
        predictions = model(img)  # detect object
        if isinstance(predictions, tuple):  # check the result is tuple
            predictions = predictions[0]
        results = non_max_suppression(predictions)  # organize the result nvms

    # draw result
    for det in results:
        if det is not None:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()  # rescale the coordinates to original image
            for *xyxy, conf, cls in det:
                label = f"{int(cls)} {conf:.2f}"
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # output frame
    cv2.imshow('YOLO Detection', frame)

    # ESC for exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
