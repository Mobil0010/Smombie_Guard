import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import winsound
import time

# ==========================================
# 1. 모델 및 설정 로딩
# ==========================================
# (1) YOLO 모델 (물체 인식)
print("🚀 YOLO 모델 로딩 중...")
yolo_model = YOLO('yolov8m.pt')
# 0:사람, 65:리모컨, 67:폰, 73:책
target_classes = [0, 65, 67, 73] 

# (2) LSTM 모델 (행동 인식)
print("🧠 LSTM 모델 로딩 중...")
lstm_model = load_model('smombie_model.h5')
actions = ['normal', 'smombie']
seq_length = 30
seq = []

# (3) MediaPipe (자세 추정)
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# (4) 기타 변수
last_beep_time = 0

# 거리 계산 함수 (손이 박스 안에 있는지)
def is_inside_box(x, y, box):
    x1, y1, x2, y2 = box
    margin = 50 # 손이 살짝 벗어나도 인정
    return (x1 - margin) < x < (x2 + margin) and (y1 - margin) < y < (y2 + margin)

# ==========================================
# 2. 메인 실행 루프
# ==========================================
cap = cv2.VideoCapture(0)

print("✅ 최종 하이브리드 시스템 가동! (YOLO + LSTM)")
print("👉 조건: [손에 폰 있음] AND [스몸비 자세] -> 경고!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ---------------------------------------------
    # Step A: MediaPipe로 뼈대 추출 (LSTM용)
    # ---------------------------------------------
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # 기본 변수 초기화
    is_holding_phone = False
    action = "analyzing..."
    lstm_conf = 0.0

    if result.pose_landmarks:
        # 1. 뼈대 그리기
        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # 2. LSTM 데이터 전처리
        joint = np.zeros((33, 4))
        for j, lm in enumerate(result.pose_landmarks.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
        
        d = joint.flatten()
        seq.append(d)
        if len(seq) > seq_length:
            seq.pop(0)

        # 3. 주요 관절 좌표 (YOLO와 매칭용)
        h, w, _ = img.shape
        landmarks = result.pose_landmarks.landmark
        left_wrist = (int(landmarks[15].x * w), int(landmarks[15].y * h))
        right_wrist = (int(landmarks[16].x * w), int(landmarks[16].y * h))
        
        # ---------------------------------------------
        # Step B: YOLO로 물체 감지 (폰 있는지?)
        # ---------------------------------------------
        # conf를 좀 낮춰서(0.3) 폰 뒷면도 잘 잡게 함
        yolo_results = yolo_model(frame, classes=target_classes, conf=0.3, verbose=False)
        
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                
                # 사람이 아니면(폰, 리모컨, 책) 박스 검사
                if cls != 0: 
                    # 시각화 (회색 박스)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (150, 150, 150), 1)
                    
                    # 손목이 이 박스 근처에 있는지 확인!
                    held_by_left = is_inside_box(left_wrist[0], left_wrist[1], [x1, y1, x2, y2])
                    held_by_right = is_inside_box(right_wrist[0], right_wrist[1], [x1, y1, x2, y2])
                    
                    if held_by_left or held_by_right:
                        is_holding_phone = True
                        # 감지된 폰은 빨간 박스로 강조
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(img, "PHONE FOUND", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        # ---------------------------------------------
        # Step C: LSTM 행동 예측 (스몸비 자세인지?)
        # ---------------------------------------------
        if len(seq) == seq_length:
            input_data = np.expand_dims(np.array(seq), axis=0)
            y_pred = lstm_model.predict(input_data, verbose=0).squeeze()
            i_pred = int(np.argmax(y_pred))
            lstm_conf = y_pred[i_pred]
            
            if lstm_conf > 0.6: # 확신도 60% 이상일 때만 갱신
                action = actions[i_pred]

    # ---------------------------------------------
    # Step D: 최종 판단 (AND 조건)
    # ---------------------------------------------
    # 1. LSTM이 'smombie'라고 판단했고
    # 2. YOLO가 '손에 폰이 있다'고 판단했을 때
    
    status_color = (0, 255, 0) # 평화로움 (초록)
    final_decision = "SAFE"

    if action == 'smombie':
        if is_holding_phone:
            # 🚨 진짜 위험 상황!
            final_decision = "DANGER: SMOMBIE"
            status_color = (0, 0, 255) # 빨강
            
            # 테두리 효과
            cv2.rectangle(img, (0,0), (w, h), (0,0,255), 10)
            cv2.putText(img, "!!! WARNING !!!", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 4)
            
            # 소리 출력
            current_time = time.time()
            if current_time - last_beep_time > 1.0:
                winsound.Beep(1000, 500)
                last_beep_time = current_time
        else:
            # 자세는 스몸비인데 폰이 없음 (빈손)
            final_decision = "Pose: Smombie (No Phone)"
            status_color = (0, 165, 255) # 주황 (주의)
    
    # 상태 텍스트 출력
    cv2.putText(img, f"Action: {action.upper()} ({lstm_conf*100:.0f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Phone Held: {is_holding_phone}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Status: {final_decision}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.imshow('Final Hybrid Guard', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()