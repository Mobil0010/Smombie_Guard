import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # 로그 지저분한 거 숨기기

actions = ['normal', 'smombie']
seq_length = 30

# 1. 데이터 불러오기 및 전처리
print("💾 데이터 로딩 중...")
data_list = []
for action in actions:
    # dataset 폴더에서 해당 액션 이름이 들어간 파일 찾기
    for file in os.listdir('dataset'):
        if action in file and file.endswith('.npy'):
            data = np.load(os.path.join('dataset', file))
            data_list.append(data)

# 모든 데이터 합치기
if len(data_list) == 0:
    print("❌ 데이터 파일이 없습니다! 01번 파일 먼저 실행해서 데이터를 모아주세요.")
    exit()

data = np.concatenate(data_list, axis=0)

print(f"총 데이터 개수: {data.shape}")

# 2. 시퀀스 데이터 만들기 (Sliding Window)
# 30프레임씩 묶어서 학습 데이터로 변환
x_data = []
y_data = []

for i in range(len(data) - seq_length):
    # 입력: 30프레임치 좌표 데이터 (마지막 라벨값 제외)
    x_data.append(data[i:i+seq_length, :-1]) 
    # 정답: 30번째 프레임의 라벨 (0 또는 1)
    y_data.append(data[i+seq_length][-1])

x_data = np.array(x_data)
y_data = to_categorical(y_data, num_classes=len(actions)) # 원-핫 인코딩

print(f"학습용 데이터셋 모양: {x_data.shape}, 정답 모양: {y_data.shape}")

# 3. 모델 구성 (LSTM)
model = Sequential([
    LSTM(64, activation='relu', input_shape=x_data.shape[1:3]), # LSTM 층
    Dense(32, activation='relu'),
    Dense(len(actions), activation='softmax') # 결과 (Normal vs Smombie)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 학습 시작
print("🔥 학습 시작!")
history = model.fit(
    x_data,
    y_data,
    epochs=30,     # 30번 반복 학습
    batch_size=32
)

# 5. 모델 저장
model.save('smombie_model.h5')
print("🎉 모델 저장 완료: smombie_model.h5")