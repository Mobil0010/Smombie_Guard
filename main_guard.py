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

print("🚀 스몸비 가드 (실시간 수치 모니터링 모드) 가동...")

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

base_pitch = 0 
base_neck = 90
base_chin_dist = 0.5 
base_low_angle = 0

# 🌟 기준값 (이 값을 넘으면 빨간불!)
THRESH_PITCH = 60      
THRESH_LOW_ANGLE = 38  
THRESH_NECK = 40       

# 스무딩 변수
smooth_pitch = 0
smooth_low = 0
smooth_neck = 0
state_head_down = False 

def dynamic_smooth(current, prev, sensitivity=2.0):
    diff = abs(current - prev)
    alpha = 0.05 if diff < sensitivity else 0.2 
    return (alpha * current) + ((1 - alpha) * prev)

def is_hand_near_box(landmarks, img_w, img_h, box, margin=100):
    x1, y1, x2, y2 = box
    hand_indices = [15, 16, 17, 18, 19, 20, 21, 22]
    hits = 0
    for idx in hand_indices:
        lx = int(landmarks[idx].x * img_w)
        ly = int(landmarks[idx].y * img_h)
        if (x1 - margin < lx < x2 + margin) and (y1 - margin < ly < y2 + margin):
            hits += 1
    return hits > 0

print(f"✅ 감시 시작! (왼쪽의 수치를 확인하세요)")

while True:
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape

    # 전처리
    enhanced_frame = utils.apply_clahe(frame)
    brightness = utils.calculate_brightness(frame)
    if brightness < 50:
        enhanced_frame = utils.apply_night_vision(enhanced_frame, gamma=2.0)
        cv2.putText(enhanced_frame, f"NIGHT MODE", (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    img_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
    
    face_results = face_mesh.process(img_rgb)
    pose_results = pose.process(img_rgb)
    
    is_phone_in_hand = False 
    is_walking = False
    
    # A. 걷기 감지
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(enhanced_frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        landmarks = pose_results.pose_landmarks.landmark
        avg_shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
        shoulder_history.append(avg_shoulder_y)
        
        if len(shoulder_history) >= 10:
            if (max(shoulder_history) - min(shoulder_history)) > WALKING_THRESHOLD:
                is_walking = True
                cv2.putText(enhanced_frame, "WALKING", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(enhanced_frame, "STANDING", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    # B. 수치 계산 (화면에 띄우기 위해 미리 계산)
    current_score = 0
    
    # [1] Pose 기반 계산
    if pose_results.pose_landmarks:
        # Neck
        raw_neck = utils.get_neck_angle(pose_results.pose_landmarks.landmark, w, h)
        raw_neck_diff = abs(raw_neck - base_neck)
        smooth_neck = dynamic_smooth(raw_neck_diff, smooth_neck)
        
        # Low Angle
        raw_low = utils.check_low_angle_score(pose_results.pose_landmarks.landmark, w, h)
        raw_low_diff = raw_low - base_low_angle
        smooth_low = dynamic_smooth(raw_low_diff, smooth_low)

        if smooth_neck > THRESH_NECK: current_score += 1
        if smooth_low > THRESH_LOW_ANGLE: current_score += 1

    # [2] FaceMesh 기반 계산
    if face_results.multi_face_landmarks:
        for fl in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(enhanced_frame, fl, mp_face_mesh.FACEMESH_TESSELATION, None, mp_drawing_styles.get_default_face_mesh_tesselation_style())

            raw_pitch, _ = utils.get_head_pose(enhanced_frame, fl.landmark)
            raw_pitch_diff = raw_pitch - base_pitch
            smooth_pitch = dynamic_smooth(raw_pitch_diff, smooth_pitch)
            
            if smooth_pitch > THRESH_PITCH: current_score += 1

            if pose_results.pose_landmarks:
                chin_dist = utils.get_chin_shoulder_distance(fl.landmark, pose_results.pose_landmarks.landmark, w, h)
                if chin_dist < (base_chin_dist * 0.8): current_score += 1

    # 🌟 [핵심] 실시간 수치 대시보드 (HUD)
    # 기준을 넘으면 빨간색(0,0,255), 안 넘으면 초록색(0,255,0)
    color_pitch = (0, 0, 255) if smooth_pitch > THRESH_PITCH else (0, 255, 0)
    color_low = (0, 0, 255) if smooth_low > THRESH_LOW_ANGLE else (0, 255, 0)
    color_neck = (0, 0, 255) if smooth_neck > THRESH_NECK else (0, 255, 0)

    # 화면 왼쪽에 출력
    cv2.putText(enhanced_frame, f"Pitch: {int(smooth_pitch)} (Limit:{THRESH_PITCH})", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_pitch, 2)
    cv2.putText(enhanced_frame, f"Low  : {int(smooth_low)} (Limit:{THRESH_LOW_ANGLE})", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_low, 2)
    cv2.putText(enhanced_frame, f"Neck : {int(smooth_neck)} (Limit:{THRESH_NECK})", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_neck, 2)

    # 히스테리시스
    if state_head_down:
        if current_score == 0 and smooth_pitch < (THRESH_PITCH - 5) and smooth_low < (THRESH_LOW_ANGLE - 5):
            state_head_down = False
    else:
        if current_score > 0:
            state_head_down = True

    if state_head_down:
        cv2.putText(enhanced_frame, "HEAD DOWN", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(enhanced_frame, "HEAD UP", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # C. 폰 감지
    yolo_results = model(enhanced_frame, classes=target_classes, conf=0.15, verbose=False)
    
    for r in yolo_results:
        for box in r.boxes:
            if int(box.cls[0]) == 0: continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            phone_box = [x1, y1, x2, y2]
            conf = float(box.conf[0])
            cls_name = model.names[int(box.cls[0])]

            is_holding = False
            if pose_results.pose_landmarks:
                is_holding = is_hand_near_box(pose_results.pose_landmarks.landmark, w, h, phone_box)
            
            if is_holding:
                is_phone_in_hand = True
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(enhanced_frame, f"{cls_name} (HELD)", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            else:
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(enhanced_frame, f"{cls_name}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # D. 최종 경고
    if is_phone_in_hand and state_head_down and is_walking:
        if int(time.time() * 5) % 2 == 0: 
            cv2.rectangle(enhanced_frame, (0,0), (w, h), (0,0,255), 20)
        cv2.putText(enhanced_frame, "SMOMBIE DETECTED!", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 4)
        if time.time() - last_beep_time > 1.0:
            winsound.Beep(1000, 500)
            last_beep_time = time.time()

    cv2.imshow('Realtime Stat Guard', enhanced_frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'): break
    elif key == ord(' '): 
        bp, bn, bc, bl = utils.calibrate_current(enhanced_frame, face_results, pose_results)
        base_pitch, base_neck, base_chin_dist, base_low_angle = bp, bn, bc, bl
        smooth_pitch = 0
        smooth_low = 0
        smooth_neck = 0
        print(f"🎯 보정 완료!")
        cv2.rectangle(enhanced_frame, (0,0), (w, h), (255, 255, 0), -1)
        cv2.imshow('Realtime Stat Guard', enhanced_frame)
        cv2.waitKey(100)

cap.release()
cv2.destroyAllWindows()