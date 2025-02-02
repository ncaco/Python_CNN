# 차량 및 보행자 검출 시스템 (Vehicle and Pedestrian Detection System)

## CNN과 YOLO의 연관성

### 1. CNN(Convolutional Neural Network)의 기본 원리
- CNN은 이미지 처리를 위한 딥러닝의 핵심 아키텍처입니다
- 주요 구성 요소:
  - Convolution Layer: 이미지의 특징을 추출
  - Pooling Layer: 특징을 압축하고 주요 정보 보존
  - Fully Connected Layer: 최종 분류/검출 수행

### 2. YOLO(You Only Look Once)와 CNN의 관계
- YOLO는 CNN을 기반으로 한 객체 검출 알고리즘입니다
- CNN의 발전된 형태로, 단일 신경망으로 객체 검출을 수행합니다
- 특징:
  - Backbone: CNN 기반의 특징 추출기 사용 (예: Darknet)
  - Neck: 특징 피라미드 네트워크(FPN)로 다양한 크기의 객체 검출
  - Head: 최종 객체 검출 및 분류 수행

### 3. 본 프로젝트에서의 적용
- YOLOv8 모델 사용:
  - CNN 기반의 backbone으로 이미지 특징 추출
  - 다중 스케일 특징 맵을 통한 객체 검출
  - 앵커 프리(Anchor-free) 방식으로 객체 위치 예측
- 성능 최적화:
  - GPU 가속을 통한 CNN 연산 최적화
  - FP16(반정밀도) 연산 지원
  - 배치 처리 지원

### 4. CNN의 장점이 반영된 부분
1. 특징 자동 추출
   - 수동 특징 엔지니어링 불필요
   - 계층적 특징 학습 가능

2. 위치 불변성
   - 객체의 위치에 관계없이 검출 가능
   - Translation invariance 특성

3. 파라미터 공유
   - 동일한 필터를 이미지 전체에 적용
   - 메모리 효율성 향상

### 5. 주요 기술적 특징
```python
# CNN 기반 특징 추출 (YOLOv8 내부)
1. 입력 이미지 -> CNN Backbone
2. 특징 맵 생성 -> Feature Pyramid
3. 객체 검출 -> Detection Head
```

### 6. 성능 최적화 포인트
- 모델 최적화:
  ```python
  if device == 'cuda':
      model = model.half()  # FP16 연산으로 CNN 가속
  ```
- 이미지 전처리:
  ```python
  img = cv2.resize(img, (640, 640))  # CNN 입력 크기 최적화
  ```

### 7. 향후 개선 방향
1. 더 깊은 CNN 아키텍처 적용
2. 커스텀 데이터셋 학습
3. 실시간 처리 최적화
4. 멀티 스케일 검출 강화

## 요구사항
- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLO
- CUDA (선택사항, GPU 가속용)

## 설치 방법
```bash
pip install -r requirements.txt
```

## 사용 방법
```bash
python vehicle_detection_v2.py
```
