import cv2
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
import numpy as np

# 모델 로드
device = select_device('cpu')  # GPU 사용 가능하면 '0'으로 설정
model = attempt_load('yolov7.pt', map_location=device)

# 비디오 파일 로드 (또는 웹캠 사용 시 '0' 입력)
cap = cv2.VideoCapture('data/t1.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO에 입력하기 위한 전처리
    img = cv2.resize(frame, (640, 640))  # YOLO 입력 크기로 변경
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
    img = np.ascontiguousarray(img)  # 메모리 정렬
    img = torch.from_numpy(img).to(device).float() / 255.0  # 텐서 변환 및 정규화
    img = img.unsqueeze(0)  # 배치 차원 추가 (batch_size, channels, height, width)

    # YOLO로 물체 감지
    with torch.no_grad():  # 추론 모드
        predictions = model(img)  # 모델 추론
        if isinstance(predictions, tuple):  # 결과가 tuple인지 확인
            predictions = predictions[0]
        results = non_max_suppression(predictions)  # NMS로 결과 정리

    # 결과 그리기
    for det in results:
        if det is not None:
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()  # 좌표 스케일 조정
            for *xyxy, conf, cls in det:
                label = f"{int(cls)} {conf:.2f}"
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 화면에 출력
    cv2.imshow('YOLO Detection', frame)

    # ESC 키로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
