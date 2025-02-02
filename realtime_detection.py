from ultralytics import YOLO
import cv2
import torch
import time
from collections import deque
import numpy as np

class RealtimeDetector:
    def __init__(self, source=0, model_path='yolov8n.pt'):
        """
        실시간 객체 검출기 초기화
        Args:
            source: 카메라 소스 (0: 기본 웹캠, RTSP URL도 가능)
            model_path: YOLO 모델 경로
        """
        # YOLO 모델 초기화
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        if self.device == 'cuda':
            self.model.half()
        
        # 비디오 캡처 초기화
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError("카메라를 열 수 없습니다.")
        
        # FPS 계산을 위한 변수
        self.fps_buffer = deque(maxlen=30)
        
        # 검출 대상 클래스
        self.target_classes = {
            0: 'person',     # 사람
            2: 'car',        # 자동차
            3: 'motorcycle', # 오토바이
            5: 'bus',        # 버스
            7: 'truck'       # 트럭
        }
        
        # 객체 추적을 위한 이전 프레임 정보
        self.prev_boxes = []
        self.prev_classes = []
    
    def calculate_fps(self):
        """FPS 계산"""
        if len(self.fps_buffer) > 0:
            return len(self.fps_buffer) / sum(self.fps_buffer)
        return 0
    
    def process_frame(self, frame):
        """
        단일 프레임 처리
        Args:
            frame: 처리할 프레임
        Returns:
            processed_frame: 처리된 프레임
            detections: 검출 결과
        """
        # 객체 검출
        results = self.model(frame, conf=0.3, iou=0.45)
        
        detection_count = {'person': 0, 'vehicle': 0}
        current_boxes = []
        current_classes = []
        
        # 검출 결과 처리
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls in self.target_classes:
                    # 바운딩 박스 좌표
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    current_boxes.append([x1, y1, x2, y2])
                    current_classes.append(cls)
                    
                    # 객체 종류에 따라 다른 색상 사용
                    if cls == 0:  # person
                        color = (0, 255, 0)
                        detection_count['person'] += 1
                    else:  # vehicles
                        color = (0, 0, 255)
                        detection_count['vehicle'] += 1
                    
                    # 바운딩 박스와 레이블 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{self.target_classes[cls]}: {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # FPS 표시
        fps = self.calculate_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 검출 수 표시
        cv2.putText(frame, f"Persons: {detection_count['person']}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Vehicles: {detection_count['vehicle']}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        self.prev_boxes = current_boxes
        self.prev_classes = current_classes
        
        return frame, detection_count
    
    def run(self):
        """실시간 검출 실행"""
        try:
            while True:
                start_time = time.time()
                
                # 프레임 읽기
                ret, frame = self.cap.read()
                if not ret:
                    print("프레임을 읽을 수 없습니다.")
                    break
                
                # 프레임 처리
                processed_frame, detections = self.process_frame(frame)
                
                # 결과 표시
                cv2.imshow('Realtime Detection', processed_frame)
                
                # FPS 계산
                process_time = time.time() - start_time
                self.fps_buffer.append(process_time)
                
                # 'q' 키로 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
    
    def __del__(self):
        """소멸자: 리소스 정리"""
        if hasattr(self, 'cap'):
            self.cap.release()

if __name__ == "__main__":
    # RTSP URL 예시 (실제 CCTV URL로 변경 필요)
    # rtsp_url = "rtsp://username:password@ip_address:port/stream"
    
    # 웹캠 테스트용
    detector = RealtimeDetector(source=0)  # 웹캠 사용
    # detector = RealtimeDetector(source=rtsp_url)  # RTSP 스트림 사용
    
    detector.run() 