import cv2
import mediapipe as mp
import numpy as np
import time
import os

# 저장할 행동 이름 (0: Normal, 1: Smombie)
actions = ['normal', 'smombie']
seq_length = 30  # 30프레임(약 1초)을 하나의 데이터 뭉치로 봄
secs_for_action = 30 # 액션당 30초 동안 녹화한다고 가정 (넉넉하게)

# MediaPipe 설정
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, frame = cap.read()
        
        print(f'=== 3초 뒤에 [{action}] 데이터 수집을 시작합니다! 준비하세요! ===')
        cv2.imshow('img', frame)
        cv2.waitKey(3000) # 3초 대기

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.pose_landmarks:
                # 33개 랜드마크의 x, y, z, visibility를 1차원으로 쫙 펼침 (33 * 4 = 132개 데이터)
                joint = np.zeros((33, 4))
                for j, lm in enumerate(result.pose_landmarks.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                
                # 각도 계산 같은 거 안 함! 그냥 좌표값(Raw Data) 그대로 넣음 -> AI가 알아서 패턴 찾음
                v = joint.flatten()
                
                # 정답 라벨(idx)을 맨 앞에 붙여줌
                v_full = np.concatenate([v, [idx]])
                data.append(v_full)

                mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break
            
            # 한 동작당 데이터를 충분히 모으면 멈춤 (여기서는 임의로 키 입력 대신 시간제한/데이터 개수로 함)
            # 0번(Normal) 끝나면 잠깐 쉬었다가 1번(Smombie) 넘어감
            if time.time() - start_time > secs_for_action:
                break
        
        data = np.array(data)
        print(f'[{action}] 데이터 수집 완료! 개수: {len(data)}')
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

    break

cap.release()
cv2.destroyAllWindows()