import cv2
import numpy as np
import math

# [1] 정면용: 3D 얼굴 각도 (PnP)
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

# [2] 옆/뒤용: 목 꺾임 각도
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

# [3] 아래쪽용: 코 vs 귀 높이 역전 (Nose-Ear)
def check_low_angle_status(landmarks, w, h):
    # Pose 랜드마크 기준: 코(0), 왼귀(7), 오른귀(8) -> (MediaPipe 버전에 따라 0이 코임)
    nose_y = landmarks[0].y * h 
    ear_y = min(landmarks[7].y * h, landmarks[8].y * h)
    return nose_y - ear_y # 양수면 코가 아래(숙임)

# [4] 위쪽용: 턱 vs 어깨 거리 (Chin-Shoulder)
def get_chin_shoulder_distance(face_landmarks, pose_landmarks, w, h):
    # FaceMesh 턱 끝: 152번
    chin = face_landmarks[152]
    chin_coords = np.array([chin.x * w, chin.y * h])
    
    # Pose 어깨 중간점
    l_sh = pose_landmarks[11]
    r_sh = pose_landmarks[12]
    shoulder_center = np.array([(l_sh.x + r_sh.x)*w/2, (l_sh.y + r_sh.y)*h/2])
    
    # 거리 계산
    distance = np.linalg.norm(chin_coords - shoulder_center)
    
    # 정규화 (사람이 멀리 있을 수도 있으니까 어깨 너비로 나눔)
    shoulder_width = abs(l_sh.x - r_sh.x) * w
    if shoulder_width == 0: return 0
    
    # 어깨 너비 대비 턱 거리가 얼마나 가까운가? (작을수록 숙인 것)
    ratio = distance / shoulder_width
    return ratio

# [보정 함수] 현재 상태를 기준값으로 저장
def calibrate_current(frame, face_results, pose_results):
    pitch = 0
    neck_angle = 90
    chin_ratio = 1.0
    
    h, w, _ = frame.shape
    
    if face_results.multi_face_landmarks:
        for fl in face_results.multi_face_landmarks:
            pitch, _ = get_head_pose(frame, fl.landmark)
            if pose_results.pose_landmarks:
                chin_ratio = get_chin_shoulder_distance(fl.landmark, pose_results.pose_landmarks.landmark, w, h)
            
    if pose_results.pose_landmarks:
        neck_angle = get_neck_angle(pose_results.pose_landmarks.landmark, w, h)
        
    return pitch, neck_angle, chin_ratio

def calculate_brightness(image):
    """
    현재 화면의 평균 밝기를 계산 (0~255)
    0에 가까울수록 암흑, 255에 가까울수록 눈뽕
    """
    # 이미지를 흑백(Grayscale)으로 변환해서 평균을 구함
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness

def apply_night_vision(image, gamma=1.5):
    """
    감마 보정(Gamma Correction)을 통해 어두운 곳을 밝게 만듦.
    gamma > 1.0 : 밝아짐
    gamma < 1.0 : 어두워짐
    """
    # 룩업 테이블(Look-Up Table) 생성 (계산 속도 100배 향상 기법)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # 이미지에 테이블 적용 (마법처럼 밝아짐!)
    return cv2.LUT(image, table)