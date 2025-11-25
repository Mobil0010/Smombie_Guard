import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import time
import winsound
import numpy as np
from collections import deque
from ultralytics import YOLO
import head_pose_utils as utils 

# ==========================================
# 1. 설정
# ==========================================
print("🚀 스몸비 가드 (노트북/로우앵글 최적화) 가동...")

model = YOLO('yolov8n.pt') 
target_classes = [0, 67, 73, 65, 77] 

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened(): exit()

shoulder_history = deque(maxlen=20)
WALKING_THRESHOLD = 0.02 
last_beep_time = 0

# 기준값 초기화
base_pitch = 0 
base_neck = 90
base_chin_dist = 0.5 
base_low_angle = 0 # 🌟 추가: 로우 앵글 기준값

print("✅ 감시 시작! (모니터 보고 [Space] 눌러서 보정 필수!)")

while True:
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape

    # 야간 모드
    brightness = utils.calculate_brightness(frame)
    if brightness < 50: 
        enhanced_frame = utils.apply_night_vision(frame, gamma=2.0)
        cv2.putText(enhanced_frame, f"NIGHT MODE ({int(brightness)})", (10, h - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        enhanced_frame = frame.copy() 

    img_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
    
    face_results = face_mesh.process(img_rgb)
    pose_results = pose.process(img_rgb)
    
    is_looking_down = False
    is_phone_in_hand = False 
    is_walking = False
    
    # A. 걷기 감지
    if pose_results.pose_landmarks:
        # 뼈대 그리기
        mp_drawing.draw_landmarks(enhanced_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        landmarks = pose_results.pose_landmarks.landmark
        avg_shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
        shoulder_history.append(avg_shoulder_y)
        
        if len(shoulder_history) >= 10:
            if (max(shoulder_history) - min(shoulder_history)) > WALKING_THRESHOLD:
                is_walking = True
                cv2.putText(enhanced_frame, "WALKING", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(enhanced_frame, "STANDING", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    # B. 고개 숙임 감지 (상대평가 적용)
    score = 0 
    debug_vals = []
    
    if pose_results.pose_landmarks:
        # [1] 목 꺾임 (측면/후방)
        neck = utils.get_neck_angle(pose_results.pose_landmarks.landmark, w, h)
        if abs(neck - base_neck) > 20: 
            score += 1
            debug_vals.append("Neck")

        # [2] 🌟 로우 앵글 보정 (핵심!)
        # 절대값이 아니라 (현재 - 기준) 차이로 계산함
        current_low = utils.check_low_angle_score(pose_results.pose_landmarks.landmark, w, h)
        low_diff = current_low - base_low_angle
        
        # 차이가 25픽셀 이상 나면 (평소보다 코가 더 내려가면) 숙임 판정
        if low_diff > 25: 
            score += 1
            debug_vals.append(f"Low({int(low_diff)})")

    if face_results.multi_face_landmarks:
        for fl in face_results.multi_face_landmarks:
            # [3] 정면 각도
            pitch, _ = utils.get_head_pose(enhanced_frame, fl.landmark)
            if (pitch - base_pitch) > 15: 
                score += 1
                debug_vals.append("Pitch")
            
            # [4] 하이 앵글
            if pose_results.pose_landmarks:
                chin_dist = utils.get_chin_shoulder_distance(fl.landmark, pose_results.pose_landmarks.landmark, w, h)
                if chin_dist < (base_chin_dist * 0.8): 
                    score += 1
                    debug_vals.append("Chin")

    if score > 0:
        is_looking_down = True
        cv2.putText(enhanced_frame, f"HEAD DOWN: {','.join(debug_vals)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(enhanced_frame, "HEAD UP", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # C. 폰 감지
    detect_conf = 0.15 
    yolo_results = model(enhanced_frame, classes=target_classes, conf=detect_conf, verbose=False)
    
    for r in yolo_results:
        for box in r.boxes:
            if int(box.cls[0]) == 0: continue 
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            phone_box = [x1, y1, x2, y2]
            conf = float(box.conf[0])
            
            is_holding = False
            if pose_results.pose_landmarks:
                is_holding = utils.is_hand_holding_phone(pose_results.pose_landmarks.landmark, w, h, phone_box)
            
            if is_holding:
                is_phone_in_hand = True
                box_color = (0, 165, 255) if conf < 0.5 else (0, 0, 255)
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), box_color, 3)
                label = f"{model.names[int(box.cls[0])]} ({conf:.2f})"
                cv2.putText(enhanced_frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            else:
                if conf > 0.5:
                    cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

    # D. 최종 경고
    if is_phone_in_hand and is_looking_down and is_walking:
        if int(time.time() * 5) % 2 == 0: 
            cv2.rectangle(enhanced_frame, (0,0), (w, h), (0,0,255), 20)
        cv2.putText(enhanced_frame, "SMOMBIE DETECTED!", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 4)
        if time.time() - last_beep_time > 1.0:
            winsound.Beep(1000, 500)
            last_beep_time = time.time()

    cv2.imshow('Laptop Optimized Guard', enhanced_frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'): break
    elif key == ord(' '): 
        # 🌟 보정할 때 로우앵글 값(bl)도 같이 저장!
        bp, bn, bc, bl = utils.calibrate_current(enhanced_frame, face_results, pose_results)
        base_pitch = bp
        base_neck = bn
        base_chin_dist = bc
        base_low_angle = bl # 기준값 업데이트
        print(f"🎯 보정 완료! LowAngle 기준: {base_low_angle:.1f}")
        cv2.rectangle(enhanced_frame, (0,0), (w, h), (255, 255, 0), -1)
        cv2.imshow('Laptop Optimized Guard', enhanced_frame)
        cv2.waitKey(100)

cap.release()
cv2.destroyAllWindows()