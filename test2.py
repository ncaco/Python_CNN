import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import matplotlib.pyplot as plt

def load_and_preprocess_image(img_path):
    # 이미지 로드
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 원본 이미지 저장
    original_img = img.copy()
    
    # MobileNetV2용 전처리
    img_resized = cv2.resize(img, (224, 224))
    x = preprocess_input(np.expand_dims(img_resized, axis=0))
    
    return original_img, x

def generate_cam(model, preprocessed_img, class_idx):
    # 마지막 컨볼루션 레이어 가져오기
    last_conv_layer = model.get_layer('Conv_1')
    
    # 모델 생성
    grad_model = tf.keras.Model(
        [model.inputs], 
        [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(preprocessed_img)
        class_channel = predictions[:, class_idx]

    # 그래디언트 계산 및 정규화
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # CAM 계산
    conv_output = conv_output[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    
    # 히트맵 정규화 및 후처리
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = tf.math.pow(heatmap, 2)  # 강조 효과
    
    return heatmap.numpy()

def process_heatmap(heatmap, original_img):
    # 히트맵 크기 조정 및 처리
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    # 히트맵 개선
    heatmap = cv2.GaussianBlur(heatmap, (9, 9), 0)  # 블러 강화
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    
    # CLAHE 적용하여 대비 개선
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    heatmap = clahe.apply(heatmap)
    
    # 컬러맵 적용
    colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    return heatmap, colored_heatmap

def detect_vehicles(gray_heatmap, original_img):
    # 이미지 전처리 개선
    gray_heatmap = cv2.normalize(gray_heatmap, None, 0, 255, cv2.NORM_MINMAX)
    gray_heatmap = cv2.convertScaleAbs(gray_heatmap)
    
    # 이미지 선명도 개선
    gray_heatmap = cv2.GaussianBlur(gray_heatmap, (5, 5), 0)
    gray_heatmap = cv2.addWeighted(gray_heatmap, 2.0, gray_heatmap, -1.0, 0)  # 대비 강화
    
    # 이진화 임계값 조정
    thresh_val = cv2.mean(gray_heatmap)[0]
    _, binary = cv2.threshold(gray_heatmap, thresh_val, 255, cv2.THRESH_BINARY)
    
    # 노이즈 제거
    binary = cv2.medianBlur(binary, 5)
    
    # 모폴로지 연산 강화
    kernel_close = np.ones((7,7), np.uint8)
    kernel_open = np.ones((3,3), np.uint8)
    
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
    binary = cv2.dilate(binary, kernel_close, iterations=1)
    
    # 윤곽선 검출
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    
    # 컨투어 필터링 개선
    filtered_contours = []
    img_area = original_img.shape[0] * original_img.shape[1]
    
    # 면적 기준 조정
    min_area = 0.002 * img_area  # 최소 면적 증가
    max_area = 0.5 * img_area    # 최대 면적 감소
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            
            # 차량 종횡비 범위 조정
            if 0.4 < aspect_ratio < 2.5:  # 더 엄격한 종횡비
                # 컨투어 형상 분석
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                
                # 사각형에 가까운 형상만 선택
                if 4 <= len(approx) <= 8:
                    # 컨투어 방향 검사
                    rect = cv2.minAreaRect(contour)
                    angle = abs(rect[2])
                    if angle < 30 or angle > 60:  # 수직/수평에 가까운 방향
                        filtered_contours.append(contour)
    
    # 바운딩 박스 생성
    boxes = []
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        # 패딩 조정
        padding_ratio = min(0.1, max(0.05, w * h / (img_area * 0.01)))  # 패딩 비율 감소
        padding_x = int(w * padding_ratio)
        padding_y = int(h * padding_ratio)
        
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w = min(original_img.shape[1] - x, w + 2*padding_x)
        h = min(original_img.shape[0] - y, h + 2*padding_y)
        boxes.append([x, y, x+w, y+h])
    
    return boxes if boxes else []

def non_max_suppression(boxes, overlap_thresh=0.2):  # overlap threshold 조정
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    pick = []
    
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)  # y 좌표 기준으로 정렬
    
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / np.minimum(area[i], area[idxs[:last]])
        
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    
    return boxes[pick].astype("int")

def draw_boxes(img, boxes, color=(0, 0, 255), thickness=2):
    img_with_boxes = img.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
        # 박스 그리기
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
        
        # 박스 크기에 따라 텍스트 크기 조정
        box_width = x2 - x1
        text_size = max(0.4, min(box_width / 100, 1.0))
        
        # 차량 레이블 추가 (번호 포함)
        label = f"Vehicle #{i}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = text_size
        font_thickness = max(1, int(thickness * text_size))
        
        # 텍스트 배경 추가
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        cv2.rectangle(img_with_boxes, (x1, y1-text_height-5), (x1+text_width, y1), color, -1)
        
        # 텍스트 추가
        cv2.putText(img_with_boxes, label, (x1, y1-5), font, font_scale, (255,255,255), font_thickness)
    
    return img_with_boxes

def visualize_results(original_img, heatmap, output, num_vehicles, title='Vehicle Detection Results'):
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.figure(figsize=(15, 5))
    
    # 원본 이미지와 검출 결과
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title(f'검출 결과 (검출된 차량: {num_vehicles}대)', fontsize=12)
    plt.axis('off')
    
    # 히트맵
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('히트맵', fontsize=12)
    plt.axis('off')
    
    # 최종 결과
    plt.subplot(1, 3, 3)
    plt.imshow(output)
    plt.title(f'최종 결과 (검출된 차량: {num_vehicles}대)', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # 이미지 로드 및 모델 준비
    img_path = 'car2.jpg'
    original_img, preprocessed_img = load_and_preprocess_image(img_path)
    model = MobileNetV2(weights='imagenet')
    
    # 클래스 예측
    preds = model.predict(preprocessed_img)
    class_idx = np.argmax(preds[0])
    
    # CAM 생성 및 처리 (항상 생성)
    heatmap = generate_cam(model, preprocessed_img, class_idx)
    gray_heatmap, colored_heatmap = process_heatmap(heatmap, original_img)
    
    # ImageNet 클래스 중 차량 관련 클래스 인덱스
    vehicle_classes = [
        817,  # sports car
        511,  # convertible
        468,  # cab, hack, taxi
        751,  # racer, race car
        757,  # sedan
        867,  # trailer truck
        864,  # tow truck
        870,  # trolleybus
        724,  # police van
        779,  # station wagon
        656,  # minivan
        717,  # pickup truck
        675,  # motor scooter
        734,  # recreational vehicle
        803,  # sports car
        829,  # streetcar
        654,  # minibus
    ]
    
    # 차량이 아닌 경우 검출하지 않음
    if class_idx not in vehicle_classes:
        print("\n차량이 검출되지 않았습니다.")
        # 결과 시각화 (검출 없이)
        output = cv2.addWeighted(original_img.copy(), 0.7, colored_heatmap, 0.3, 0)
        visualize_results(original_img, colored_heatmap, output, 0)
        return
    
    # 차량 검출
    boxes = detect_vehicles(gray_heatmap, original_img)
    num_vehicles = 0
    
    if boxes:
        # NMS threshold 조정
        boxes = non_max_suppression(boxes, overlap_thresh=0.1)
        num_vehicles = len(boxes)
        original_img = draw_boxes(original_img, boxes, color=(0, 0, 255), thickness=2)
        
        # 히트맵과 원본 이미지 블렌딩 비율 조정
        output = cv2.addWeighted(original_img.copy(), 0.7, colored_heatmap, 0.3, 0)
        output = draw_boxes(output, boxes, color=(0, 0, 255), thickness=2)
    else:
        output = cv2.addWeighted(original_img.copy(), 0.7, colored_heatmap, 0.3, 0)
    
    # 결과 시각화
    visualize_results(original_img, colored_heatmap, output, num_vehicles)
    
    # 콘솔에 결과 출력
    print(f"\n검출 결과: 총 {num_vehicles}대의 차량이 검출되었습니다.")

if __name__ == '__main__':
    main()
