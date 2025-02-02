import requests
import xml.etree.ElementTree as ET
from ultralytics import YOLO
import cv2
import torch
import time
from collections import deque
import numpy as np
import json
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

class HighwayCCTVDetector:
    def __init__(self, api_key):
        """
        고속도로 CCTV 객체 검출기 초기화
        Args:
            api_key: 국가교통정보센터 OpenAPI 인증키
        """
        self.api_key = api_key
        self.base_url = "http://openapi.its.go.kr"
        self.rtsp_base = "rtsp://openapi.its.go.kr:554"
        
        # YOLO 모델 초기화
        self.model = YOLO('yolov8n.pt')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        if self.device == 'cuda':
            self.model.half()
        
        # 검출 대상 클래스
        self.target_classes = {
            0: 'person',     # 사람
            2: 'car',        # 자동차
            3: 'motorcycle', # 오토바이
            5: 'bus',        # 버스
            7: 'truck'       # 트럭
        }
        
        self.fps_buffer = deque(maxlen=30)
    
    def get_cctv_list(self, route_no=None):
        """
        고속도로 CCTV 목록 조회
        Args:
            route_no: 고속도로 노선번호 (선택)
        Returns:
            cctv_list: CCTV 정보 리스트
        """
        params = {
            'key': self.api_key,
            'type': 'json',
            'cctvType': '1',  # 실시간 스트리밍 가능 CCTV
            'pageSize': '100',
            'version': '2'
        }
        
        if route_no:
            params['routeNo'] = route_no
        
        # 최대 3번 재시도    
        for attempt in range(3):
            try:
                response = requests.get(
                    f"{self.base_url}/api/cctvInfo", 
                    params=params,
                    timeout=10,  # 10초 타임아웃
                    verify=False  # SSL 인증서 검증 비활성화
                )
                response.raise_for_status()
                
                data = response.json()
                return data.get('response', {}).get('data', [])
            except requests.Timeout:
                print(f"시도 {attempt + 1}/3: 요청 시간 초과")
                if attempt < 2:  # 마지막 시도가 아니면 재시도
                    time.sleep(2)  # 2초 대기 후 재시도
                continue
            except Exception as e:
                print(f"CCTV 목록 조회 실패: {e}")
                if attempt < 2:
                    time.sleep(2)
                continue
        return []
    
    def get_cctv_url(self, cctv_id):
        """
        CCTV 스트리밍 URL 조회
        Args:
            cctv_id: CCTV ID
        Returns:
            rtsp_url: RTSP 스트리밍 URL
        """
        params = {
            'key': self.api_key,
            'type': 'json',
            'cctvId': cctv_id,
            'version': '2'
        }
        
        # 최대 3번 재시도
        for attempt in range(3):
            try:
                response = requests.get(
                    f"{self.base_url}/api/cctvStream", 
                    params=params,
                    timeout=10,  # 10초 타임아웃
                    verify=False  # SSL 인증서 검증 비활성화
                )
                response.raise_for_status()
                
                data = response.json()
                url_info = data.get('response', {}).get('streamUrl', '')
                return f"{self.rtsp_base}{url_info}" if url_info else None
            except requests.Timeout:
                print(f"시도 {attempt + 1}/3: 요청 시간 초과")
                if attempt < 2:
                    time.sleep(2)
                continue
            except Exception as e:
                print(f"스트리밍 URL 조회 실패: {e}")
                if attempt < 2:
                    time.sleep(2)
                continue
        return None
    
    def process_stream(self, cctv_id, save_video=False):
        """
        CCTV 스트림 처리
        Args:
            cctv_id: CCTV ID
            save_video: 비디오 저장 여부
        """
        rtsp_url = self.get_cctv_url(cctv_id)
        if not rtsp_url:
            print("스트리밍 URL을 가져올 수 없습니다.")
            return
        
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            print("CCTV 스트림을 열 수 없습니다.")
            return
        
        # 비디오 저장 설정
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(f'cctv_{cctv_id}_{time.strftime("%Y%m%d_%H%M%S")}.mp4',
                                fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("프레임을 읽을 수 없습니다.")
                    break
                
                # 객체 검출
                results = self.model(frame, conf=0.3, iou=0.45)
                
                detection_count = {'person': 0, 'vehicle': 0}
                
                # 검출 결과 처리
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        if cls in self.target_classes:
                            # 바운딩 박스 좌표
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
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
                
                # FPS 계산 및 표시
                process_time = time.time() - start_time
                self.fps_buffer.append(process_time)
                fps = len(self.fps_buffer) / sum(self.fps_buffer) if self.fps_buffer else 0
                
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Vehicles: {detection_count['vehicle']}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 결과 표시
                cv2.imshow(f'Highway CCTV {cctv_id}', frame)
                
                if save_video:
                    out.write(frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            if save_video:
                out.release()
            cv2.destroyAllWindows()

def main():
    # API 키 설정 (실제 발급받은 키로 변경 필요)
    API_KEY = "3770717658"
    
    # requests의 경고 메시지 비활성화
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    # 검출기 초기화
    detector = HighwayCCTVDetector(API_KEY)
    
    # 고속도로 CCTV 목록 조회 (예: 경부고속도로 = 0010)
    print("CCTV 목록을 조회하는 중...")
    cctv_list = detector.get_cctv_list(route_no='0010')
    
    if cctv_list:
        print("\n사용 가능한 CCTV 목록:")
        for cctv in cctv_list:
            print(f"ID: {cctv.get('cctvId')}, 위치: {cctv.get('cctvName')}")
        
        # 첫 번째 CCTV로 테스트
        first_cctv = cctv_list[0].get('cctvId')
        print(f"\n{first_cctv} CCTV 스트림을 시작합니다...")
        detector.process_stream(first_cctv, save_video=True)
    else:
        print("\n사용 가능한 CCTV가 없습니다.")
        print("1. API 키가 올바른지 확인해주세요")
        print("2. 네트워크 연결을 확인해주세요")
        print("3. 잠시 후 다시 시도해주세요")

if __name__ == "__main__":
    main() 