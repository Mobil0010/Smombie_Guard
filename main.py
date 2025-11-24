import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque
import winsound  # 윈도우 경고음용
import math
import time

# ==========================================
# 1. 설정 및 모델 로딩
# ==========================================
model = YOLO('yolov8m.pt') 
target_classes = [0, 65, 67, 73] # 사람, 리모컨, 폰, 책

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 걷기 감지용 데이터 저장소 (최근 30프레임 어깨 높이 저장)
shoulder_history = deque(maxlen=30)

# 경고 쿨타임 (소리가 너무 연속으로 나면 시끄러우니까)
last_beep_time = 0

# ==========================================
# 2. 유틸리티 함수 (각도 & 계산)
# ==========================================
def calculate_angle(a, b):
    # 두 점 사이의 각도(Y축 기준) 계산
    # a: 어깨, b: 귀
    a = np.array(a)
    b = np.array(b)
    
    radians = np.arctan2(b[1] - a[1], b[0] - a[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    # 수직 기준 각도로 변환 (90도가 정자세라고 가정 시)
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def is_walking(history, threshold=0.005):
    # 어깨 높이의 '최대값 - 최소값' 차이가 크면 움직이는 중!
    if len(history) < 10:
        return False
    diff = max(history) - min(history)
    return diff > threshold

def is_inside_box(x, y, box):
    x1, y1, x2, y2 = box
    margin = 40
    return (x1 - margin) < x < (x2 + margin) and (y1 - margin) < y < (y2 + margin)

# ==========================================
# 3. 메인 실행
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 웹캠 연결 실패!")
    exit()

print("✅ Smombie Guard 최종판: [폰 들기 + 고개 숙임 + 걷기] 감지 중...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 기본 준비
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)
    yolo_results = model(frame, classes=target_classes, conf=0.35, verbose=False)

    # 2. 상태 플래그 (초기화)
    cond_phone_in_hand = False
    cond_head_down = False
    cond_walking = False

    # -------------------------------------------------
    # [Step 1] YOLO + 손 위치 (핸드폰 들었나?)
    # -------------------------------------------------
    phone_boxes = []
    for r in yolo_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            if cls != 0: # 사람이 아니면(폰, 리모컨, 책) 박스 저장
                phone_boxes.append([x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 150, 150), 1)

    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark

        # 주요 좌표 추출 (어깨, 귀, 손목)
        left_shoulder = [landmarks[11].x * w, landmarks[11].y * h]
        right_shoulder = [landmarks[12].x * w, landmarks[12].y * h]
        left_ear = [landmarks[7].x * w, landmarks[7].y * h]
        right_ear = [landmarks[8].x * w, landmarks[8].y * h]
        left_wrist = [landmarks[15].x * w, landmarks[15].y * h]
        right_wrist = [landmarks[16].x * w, landmarks[16].y * h]
        
        # 배꼽(Hip) 높이 계산 (손을 들었는지 판단용)
        avg_hip_y = (landmarks[23].y * h + landmarks[24].y * h) / 2

        # -------------------------------------------------
        # [Step 2] 고개 각도 계산 (숙였나?)
        # -------------------------------------------------
        # 어깨와 귀를 잇는 각도 계산
        neck_angle_left = calculate_angle(left_shoulder, left_ear)
        neck_angle_right = calculate_angle(right_shoulder, right_ear)
        
        # 각도가 낮을수록 고개를 앞으로 내민 것 (수직에 가까우면 90도 근처)
        # 70도 미만이면 거북목/숙임으로 판단 (테스트하며 조절 필요!)
        NECK_THRESHOLD = 70 
        
        if neck_angle_left < NECK_THRESHOLD or neck_angle_right < NECK_THRESHOLD:
            cond_head_down = True
            cv2.putText(frame, "HEAD DOWN", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # -------------------------------------------------
        # [Step 3] 걷기 감지 (움직이나?)
        # -------------------------------------------------
        # 양쪽 어깨의 Y좌표 평균을 히스토리에 저장
        avg_shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
        shoulder_history.append(avg_shoulder_y)

        if is_walking(shoulder_history, threshold=0.015): # 민감도 조절 (숫자가 작으면 민감)
            cond_walking = True
            cv2.putText(frame, "WALKING", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # -------------------------------------------------
        # [Step 4] 폰 소지 여부 (손목 + 박스)
        # -------------------------------------------------
        for box in phone_boxes:
            held_by_left = is_inside_box(left_wrist[0], left_wrist[1], box)
            held_by_right = is_inside_box(right_wrist[0], right_wrist[1], box)
            is_hands_up = (left_wrist[1] < avg_hip_y) or (right_wrist[1] < avg_hip_y)

            if (held_by_left or held_by_right) and is_hands_up:
                cond_phone_in_hand = True
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)

    # ==========================================
    # 🚨 최종 판단 및 경고
    # ==========================================
    status_text = f"Phone:{int(cond_phone_in_hand)} | Head:{int(cond_head_down)} | Walk:{int(cond_walking)}"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 3가지 조건이 모두 True일 때만 경고!
    if cond_phone_in_hand and cond_head_down and cond_walking:
        # 화면 효과
        cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
        cv2.putText(frame, "!!! DANGER !!!", (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        
        # 소리 (삐- 소리, 1초에 한 번씩만)
        current_time = time.time()
        if current_time - last_beep_time > 1.0:
            # 윈도우 비프음 (주파수 1000Hz, 500ms 지속)
            winsound.Beep(1000, 500)
            last_beep_time = current_time

    cv2.imshow("Smombie Guard Final", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()