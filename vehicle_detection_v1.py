from ultralytics import YOLO
import cv2
import numpy as np
import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules import Conv

# YOLO 모델 로드 전에 안전한 글로벌 설정 추가
torch.serialization.add_safe_globals([DetectionModel, Sequential, Conv])

# 원본 torch.load 함수 저장
_original_load = torch.load

# torch.load 재정의
def custom_load(f, *args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(f, *args, **kwargs)

torch.load = custom_load

def detect_vehicles(image_path):
    # YOLO 모델 로드
    model = YOLO('yolov8n.pt')
    
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("이미지를 불러올 수 없습니다.")
    
    # 객체 검출 수행
    results = model(img)
    
    # 차량 클래스 ID (COCO dataset 기준)
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    vehicle_count = 0
    
    # 검출된 객체에 대해 처리
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 클래스 확인
            cls = int(box.cls[0])
            if cls in vehicle_classes:
                vehicle_count += 1
                # 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 빨간색 박스 그리기
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    print(f"검출된 차량 수: {vehicle_count}")
    return img, vehicle_count

def save_result(img, output_path):
    cv2.imwrite(output_path, img)
    print(f"결과가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    # 이미지 경로 설정
    input_image = "car.jpg"  # 분석할 이미지 경로
    output_image = "result1.jpg"     # 결과 이미지 저장 경로
    
    # 차량 검출 수행
    result_img, count = detect_vehicles(input_image)
    
    # 결과 저장
    save_result(result_img, output_image) 