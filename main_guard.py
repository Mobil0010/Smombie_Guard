import cv2
import mediapipe as mp
import time
import winsound
import numpy as np
from collections import deque
from ultralytics import YOLO
import head_pose_utils as utils 

# ==========================================
# 1. 설정 (튜닝된 값 적용)
# ==========================================
print("🚀 스몸비 가드 (최적화 버전) 가동 중...")

# 🌟 [수정 1] 속도 향상을 위해 가벼운 모델(Nano)로 교체!
# m(medium) -> n(nano) : 반응 속도가 훨씬 빨라짐
model = YOLO('yolov8m.pt') 
target_classes = [0, 67, 73] 

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened(): exit()

shoulder_history = deque(maxlen=20)

# 🌟 [수정 2] 걷기 감지 기준 대폭 완화 (0.005 -> 0.02)
# 이제 어깨를 꽤 크게 움직여야 '걷기'로 인식함 (숨쉬기는 무시)
WALKING_THRESHOLD = 0.02 
last_beep_time = 0

base_pitch = 0 
base_neck = 90
base_chin_dist = 0.5 

print("✅ 감시 시작! (반응 속도 UP, 기준 완화)")

while True:
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape # 에러 방지용 위치

    # ----------------------------------
    # 🌟 [Step 0] 야간 모드 체크 (기준 변경)
    # ----------------------------------
    brightness = utils.calculate_brightness(frame)
    
    # 🌟 [수정 3] 나이트 모드 기준 낮춤 (80 -> 50)
    # 이제 웬만한 실내 조명에서는 안 켜짐. 진짜 어두울 때만 켜짐.
    if brightness < 50: 
        enhanced_frame = utils.apply_night_vision(frame, gamma=2.0)
        cv2.putText(enhanced_frame, f"NIGHT MODE (Bright:{int(brightness)})", (10, h - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        enhanced_frame = frame.copy() 

    img_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
    
    # ----------------------------------
    # AI 모델 실행
    # ----------------------------------
    face_results = face_mesh.process(img_rgb)
    pose_results = pose.process(img_rgb)
    
    is_looking_down = False
    is_phone_detected = False
    is_walking = False
    debug_text = ""

    # A. 고개 숙임 감지
    score = 0 
    if pose_results.pose_landmarks:
        neck = utils.get_neck_angle(pose_results.pose_landmarks.landmark, w, h)
        if abs(neck - base_neck) > 20: 
            score += 1
            debug_text += "Neck "
        low_angle = utils.check_low_angle_status(pose_results.pose_landmarks.landmark, w, h)
        if low_angle > 20: 
            score += 1
            debug_text += "LowAngle "

    if face_results.multi_face_landmarks:
        for fl in face_results.multi_face_landmarks:
            pitch, _ = utils.get_head_pose(enhanced_frame, fl.landmark)
            if (pitch - base_pitch) > 15: 
                score += 1
                debug_text += "Pitch "
            if pose_results.pose_landmarks:
                chin_dist = utils.get_chin_shoulder_distance(fl.landmark, pose_results.pose_landmarks.landmark, w, h)
                if chin_dist < (base_chin_dist * 0.8): 
                    score += 1
                    debug_text += "ChinDist "

    if score > 0:
        is_looking_down = True
        cv2.putText(enhanced_frame, f"HEAD DOWN: {debug_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(enhanced_frame, "HEAD UP", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # B. 걷기 감지 (덜 예민하게)
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        avg_shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
        shoulder_history.append(avg_shoulder_y)
        
        if len(shoulder_history) >= 10:
            # 진폭(amplitude)이 0.02 이상이어야 걷는 것으로 인정
            if (max(shoulder_history) - min(shoulder_history)) > WALKING_THRESHOLD:
                is_walking = True
                cv2.putText(enhanced_frame, "WALKING", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(enhanced_frame, "STANDING", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)

    # C. 폰 감지
    detect_conf = 0.25 if brightness < 50 else 0.35
    yolo_results = model(enhanced_frame, classes=target_classes, conf=detect_conf, verbose=False)
    for r in yolo_results:
        for box in r.boxes:
            if int(box.cls[0]) != 0: 
                is_phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(enhanced_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # D. 최종 경고
    if is_phone_detected and is_looking_down and is_walking:
        cv2.rectangle(enhanced_frame, (0,0), (w, h), (0,0,255), 10)
        cv2.putText(enhanced_frame, "SMOMBIE DETECTED!", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 4)
        if time.time() - last_beep_time > 1.0:
            winsound.Beep(1000, 500)
            last_beep_time = time.time()

    cv2.imshow('Optimized Guard', enhanced_frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'): break
    elif key == ord(' '): 
        bp, bn, bc = utils.calibrate_current(enhanced_frame, face_results, pose_results)
        base_pitch = bp
        base_neck = bn
        base_chin_dist = bc
        print(f"🎯 보정 완료!")
        cv2.rectangle(enhanced_frame, (0,0), (w, h), (255, 255, 0), -1)
        cv2.imshow('Optimized Guard', enhanced_frame)
        cv2.waitKey(100)

cap.release()
cv2.destroyAllWindows()