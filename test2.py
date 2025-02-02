import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
import matplotlib.pyplot as plt

# 이미지 로드 및 전처리
img_path = 'car.jpg'  # 자동차 이미지 경로
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (224, 224))

# MobileNetV2 모델 로드
model = MobileNetV2(weights='imagenet')

# 이미지 전처리 및 예측
x = preprocess_input(np.expand_dims(img_resized, axis=0))
preds = model.predict(x)
class_idx = np.argmax(preds[0])

# 클래스 활성화 맵(CAM) 생성
last_conv_layer = model.get_layer('Conv_1')
grad_model = tf.keras.Model([model.inputs], [last_conv_layer.output, model.output])

with tf.GradientTape() as tape:
    conv_output, predictions = grad_model(x)
    loss = predictions[:, class_idx]

grads = tape.gradient(loss, conv_output)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
conv_output = conv_output[0]
heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

# 히트맵 처리 개선
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)

# 노이즈 제거를 위한 블러 처리
heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# 히트맵에서 자동차 영역 검출
gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
# Otsu's 이진화 방법 사용
_, binary = cv2.threshold(gray_heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 모폴로지 연산으로 노이즈 제거 및 영역 개선
kernel = np.ones((5,5), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 컨투어 필터링 개선
filtered_contours = []
img_area = img.shape[0] * img.shape[1]
for contour in contours:
    area = cv2.contourArea(contour)
    # 면적 기준 필터링
    if 0.1 * img_area < area < 0.4 * img_area:
        # 종횡비 검사 추가
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        # 일반적인 도로 장면의 종횡비 범위
        if 1.0 < aspect_ratio < 3.0:
            # 컨투어 형상 복잡도 검사
            perimeter = cv2.arcLength(contour, True)
            complexity = perimeter / (4 * np.sqrt(area))
            if complexity < 2.0:  # 너무 복잡한 형상 제외
                filtered_contours.append(contour)

# 가장 큰 윤곽선 찾기
if filtered_contours:
    largest_contour = max(filtered_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    # 바운딩 박스 크기 미세 조정
    padding = 10
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2*padding)
    h = min(img.shape[0] - y, h + 2*padding)
    # 붉은색 사각형 그리기
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# 원본 이미지에 히트맵 오버레이
output = cv2.addWeighted(img.copy(), 0.7, heatmap, 0.3, 0)
# 세그멘테이션 결과에도 바운딩 박스 추가
if filtered_contours:
    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)

# 파일 인코딩 설정
plt.rcParams['font.family'] = 'Malgun Gothic'

# 결과 시각화
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('원본 이미지')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(output)
plt.title('세그멘테이션 결과')
plt.axis('off')
plt.show()
