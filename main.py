import cv2
import math
import mediapipe as mp
from ultralytics import YOLO

# ==========================================
# 1. 모델 & 설정 로딩
# ==========================================
# YOLO 모델 (사람, 폰, 리모컨 감지)
model = YOLO('yolov8m.pt') 
target_classes = [0, 65, 67] # 0:사람, 65:리모컨(폰뒷면), 67:핸드폰

# MediaPipe Pose (사람 뼈대 찾기)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ==========================================
# 2. 유틸리티 함수 (거리 계산)
# ==========================================
def is_inside_box(x, y, box):
    # x, y 좌표가 네모 박스 안에 있는지 확인
    x1, y1, x2, y2 = box
    return x1 < x < x2 and y1 < y < y2

# ==========================================
# 3. 메인 실행
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ 웹캠 연결 실패!")
    exit()

print("✅ Smombie Guard v3: YOLO + MediaPipe 합체!")
print("👉 '손에 들고' + '높이 든' 물건만 폰으로 인정합니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipe는 RGB 색상을 좋아해서 변환해줘야 함
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(frame_rgb)

    # YOLO 감지
    yolo_results = model(frame, classes=target_classes, conf=0.4, verbose=False)

    # 1차적으로 감지된 폰/리모컨 박스들을 저장할 리스트
    phone_boxes = []

    # YOLO 결과 처리
    for r in yolo_results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            
            # 사람이면 흰색 박스 그냥 그려줌
            if cls == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
            
            # 폰(67)이거나 리모컨(65)이면 일단 후보군에 등록!
            elif cls == 67 or cls == 65:
                phone_boxes.append([x1, y1, x2, y2])
                # 일단 얇은 회색 박스로 표시 (아직 확정 아님)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

    # 🌟 [핵심 로직] MediaPipe랑 크로스 체크!
    if pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks.landmark
        h, w, _ = frame.shape

        # 주요 관절 좌표 가져오기 (0.0~1.0 비율이라 픽셀 좌표로 변환)
        # 15: 왼쪽 손목, 16: 오른쪽 손목
        left_wrist = (int(landmarks[15].x * w), int(landmarks[15].y * h))
        right_wrist = (int(landmarks[16].x * w), int(landmarks[16].y * h))
        
        # 23: 왼쪽 엉덩이, 24: 오른쪽 엉덩이 (높이 기준점)
        left_hip_y = int(landmarks[23].y * h)
        right_hip_y = int(landmarks[24].y * h)
        avg_hip_y = (left_hip_y + right_hip_y) // 2

        # 뼈대 그려주기 (디버깅용)
        cv2.circle(frame, left_wrist, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_wrist, 5, (0, 255, 0), -1)
        # 배꼽 라인 그리기
        cv2.line(frame, (0, avg_hip_y), (w, avg_hip_y), (0, 255, 255), 1)

        # 🧐 검증: YOLO가 찾은 박스들이 진짜 손에 들려있는지?
        for box in phone_boxes:
            bx1, by1, bx2, by2 = box
            
            # 조건 1: 손목이 박스 근처에 있는가? (확장된 박스로 체크)
            # 박스를 좀 넉넉하게(margin) 잡아서 손목이 살짝 벗어나도 인정해줌
            margin = 50
            expanded_box = [bx1 - margin, by1 - margin, bx2 + margin, by2 + margin]
            
            held_by_left = is_inside_box(left_wrist[0], left_wrist[1], expanded_box)
            held_by_right = is_inside_box(right_wrist[0], right_wrist[1], expanded_box)

            # 조건 2: 손목 높이가 배꼽(Hip)보다 높은가? (Y좌표는 위로 갈수록 작아짐!)
            # 즉, wrist_y < hip_y 여야 손을 든 것임.
            is_hands_up = (left_wrist[1] < avg_hip_y) or (right_wrist[1] < avg_hip_y)

            if (held_by_left or held_by_right) and is_hands_up:
                # 🎉 빙고! 이건 빼박 폰이다!
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
                cv2.putText(frame, "SMARTPHONE DETECTED", (bx1, by1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # 손에 없거나 손이 너무 낮으면 무시 (마우스, 지갑 등)
                pass

    cv2.imshow("Smombie Guard Step 3", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()