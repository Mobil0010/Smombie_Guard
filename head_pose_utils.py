import cv2
import numpy as np
import math

# [1] 정면용 (PnP) - 그대로
def get_head_pose(image, landmarks):
    h, w, _ = image.shape
    face_3d = np.array([
        [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
    ], dtype=np.float64)
    key_landmarks = [1, 199, 33, 263, 61, 291]
    face_2d = []
    for idx in key_landmarks:
        lm = landmarks[idx]
        face_2d.append([int(lm.x * w), int(lm.y * h)])
    face_2d = np.array(face_2d, dtype=np.float64)
    focal_length = 1 * w
    cam_matrix = np.array([[focal_length, 0, w/2], [0, focal_length, h/2], [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles[0] * 360, angles[1] * 360

# [2] 옆/뒤용 (Neck) - 그대로
def get_neck_angle(landmarks, w, h):
    l_vis, r_vis = landmarks[7].visibility, landmarks[8].visibility
    if l_vis > r_vis:
        ear = [landmarks[7].x * w, landmarks[7].y * h]
        shoulder = [landmarks[11].x * w, landmarks[11].y * h]
    else:
        ear = [landmarks[8].x * w, landmarks[8].y * h]
        shoulder = [landmarks[12].x * w, landmarks[12].y * h]
    delta_x = ear[0] - shoulder[0]
    delta_y = ear[1] - shoulder[1]
    return abs(math.degrees(math.atan2(delta_y, delta_x)))

# [3] 아래쪽용 (Low Angle) - 그대로
def check_low_angle_score(landmarks, w, h):
    # 코(0)와 귀(7,8)의 Y좌표 차이 계산
    nose_y = landmarks[0].y * h 
    ear_y = min(landmarks[7].y * h, landmarks[8].y * h)
    return nose_y - ear_y # 값이 클수록 코가 아래에 있는 것

# [4] 위쪽용 (High Angle) - 그대로
def get_chin_shoulder_distance(face_landmarks, pose_landmarks, w, h):
    chin = face_landmarks[152]
    chin_coords = np.array([chin.x * w, chin.y * h])
    l_sh = pose_landmarks[11]
    r_sh = pose_landmarks[12]
    shoulder_center = np.array([(l_sh.x + r_sh.x)*w/2, (l_sh.y + r_sh.y)*h/2])
    distance = np.linalg.norm(chin_coords - shoulder_center)
    shoulder_width = abs(l_sh.x - r_sh.x) * w
    if shoulder_width == 0: return 0
    return distance / shoulder_width

# [5] 손 그립 체크 - 그대로
def is_hand_holding_phone(landmarks, img_w, img_h, box, margin=100):
    x1, y1, x2, y2 = box
    hand_points = [15, 16, 17, 18, 19, 20, 21, 22]
    hits = 0
    thumb_pos = None
    index_pos = None
    for idx in hand_points:
        lx = int(landmarks[idx].x * img_w)
        ly = int(landmarks[idx].y * img_h)
        if (x1 - margin < lx < x2 + margin) and (y1 - margin < ly < y2 + margin):
            hits += 1
            if idx == 21 or idx == 22: thumb_pos = np.array([lx, ly])
            if idx == 19 or idx == 20: index_pos = np.array([lx, ly])
    if hits < 2: return False
    if thumb_pos is not None and index_pos is not None:
        if np.linalg.norm(thumb_pos - index_pos) < 30: return False 
    return True

# 🌟 [수정] 보정 함수: 로우 앵글 값(nose_diff)도 저장하도록 변경!
def calibrate_current(frame, face_results, pose_results):
    pitch = 0
    neck_angle = 90
    chin_ratio = 1.0
    low_angle_score = 0 # 추가됨
    
    h, w, _ = frame.shape
    
    if face_results.multi_face_landmarks:
        for fl in face_results.multi_face_landmarks:
            pitch, _ = get_head_pose(frame, fl.landmark)
            if pose_results.pose_landmarks:
                chin_ratio = get_chin_shoulder_distance(fl.landmark, pose_results.pose_landmarks.landmark, w, h)
            
    if pose_results.pose_landmarks:
        neck_angle = get_neck_angle(pose_results.pose_landmarks.landmark, w, h)
        low_angle_score = check_low_angle_score(pose_results.pose_landmarks.landmark, w, h)
        
    return pitch, neck_angle, chin_ratio, low_angle_score # 값 4개 리턴