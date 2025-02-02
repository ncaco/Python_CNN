from ultralytics import YOLO
import cv2
import numpy as np
import torch
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
from ultralytics.nn.modules import Conv

# YOLO 모델 로드시 필요한 클래스들을 안전하게 직렬화하기 위한 설정
torch.serialization.add_safe_globals([DetectionModel, Sequential, Conv])

# PyTorch의 기본 load 함수를 커스텀 함수로 대체하기 위한 설정
_original_load = torch.load

def custom_load(f, *args, **kwargs):
    """
    PyTorch의 모델 로드 함수를 커스터마이즈
    weights_only=False로 설정하여 전체 모델 구조를 로드
    """
    kwargs['weights_only'] = False
    return _original_load(f, *args, **kwargs)

torch.load = custom_load

def preprocess_image(img):
    """
    입력 이미지 전처리 함수
    Args:
        img: 입력 이미지
    Returns:
        전처리된 이미지 (640x640 크기로 조정)
    """
    img = cv2.resize(img, (640, 640))
    return img

def detect_vehicles(image_path):
    """
    이미지에서 차량과 사람을 검출하는 메인 함수
    Args:
        image_path: 분석할 이미지 경로
    Returns:
        original_img: 검출 결과가 표시된 이미지
        detection_count: 검출된 객체 수 카운트
    """
    # YOLO 모델 초기화 및 GPU 설정
    model = YOLO('yolov8n.pt')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    if device == 'cuda':
        model = model.half()  # GPU 사용시 FP16 연산으로 속도 향상
    
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("이미지를 불러올 수 없습니다.")
    
    # 원본 이미지 보존
    original_img = img.copy()
    
    # 객체 검출 수행 (신뢰도 0.3, IoU 임계값 0.45)
    results = model(img, conf=0.3, iou=0.45)
    
    # COCO 데이터셋 기준 검출 대상 클래스 정의
    target_classes = {
        0: 'person',     # 사람
        2: 'car',        # 자동차
        3: 'motorcycle', # 오토바이
        5: 'bus',        # 버스
        7: 'truck'       # 트럭
    }
    
    # 검출 결과 저장을 위한 변수 초기화
    detection_count = {'person': 0, 'vehicle': 0}
    detected_boxes = []  # 중복 검출 방지를 위한 리스트
    
    # 검출된 모든 객체에 대해 처리
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])    # 클래스 ID
            conf = float(box.conf[0]) # 신뢰도 점수
            
            # 지정된 클래스에 대해서만 처리
            if cls in target_classes:
                # 현재 검출된 박스의 좌표
                current_box = box.xyxy[0].cpu().numpy()
                is_duplicate = False
                
                # 중복 검출 확인 (IoU 기반)
                for detected_box in detected_boxes:
                    iou = calculate_iou(current_box, detected_box)
                    if iou > 0.45:  # IoU가 0.45 이상이면 중복으로 판단
                        is_duplicate = True
                        break
                
                # 중복이 아닌 경우에만 처리
                if not is_duplicate:
                    # 사람과 차량을 다른 색상으로 표시
                    if cls == 0:
                        detection_count['person'] += 1
                        color = (0, 255, 0)  # 사람은 초록색
                    else:
                        detection_count['vehicle'] += 1
                        color = (0, 0, 255)  # 차량은 빨간색
                    
                    detected_boxes.append(current_box)
                    
                    # 검출 결과 시각화
                    x1, y1, x2, y2 = map(int, current_box)
                    cv2.rectangle(original_img, (x1, y1), (x2, y2), color, 2)
                    
                    # 클래스명과 신뢰도 표시
                    label = f"{target_classes[cls]}: {conf:.2f}"
                    cv2.putText(original_img, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 검출 결과 출력
    print(f"검출된 사람 수: {detection_count['person']}")
    print(f"검출된 차량 수: {detection_count['vehicle']}")
    return original_img, detection_count

def calculate_iou(box1, box2):
    """
    두 바운딩 박스 간의 IoU(Intersection over Union) 계산
    Args:
        box1, box2: [x1, y1, x2, y2] 형식의 바운딩 박스 좌표
    Returns:
        float: IoU 값 (0~1)
    """
    # 교집합 영역 계산
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

def save_result(img, output_path):
    """
    검출 결과가 표시된 이미지를 파일로 저장
    Args:
        img: 저장할 이미지
        output_path: 저장 경로
    """
    cv2.imwrite(output_path, img)
    print(f"결과가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    # 메인 실행 부분
    input_image = "car.jpg"    # 분석할 이미지 경로
    output_image = "result1.jpg"  # 결과 이미지 저장 경로
    
    # 객체 검출 수행
    result_img, counts = detect_vehicles(input_image)
    
    # 결과 저장
    save_result(result_img, output_image) 