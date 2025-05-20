import sounddevice as sd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import time

# --- YAMNetの読み込みとラベル準備 ---
model = hub.load("https://tfhub.dev/google/yamnet/1")
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
with open(class_map_path) as f:
    class_names = [line.strip().split(',')[2] for line in f.readlines()[1:]]
laughter_index = class_names.index("Laughter")

# --- サンプリング設定 ---
sr = 16000  # YAMNetの前提
duration = 1  # 1秒ごとに処理
laughter_scores = []

# --- リアルタイム処理 ---
print("🎤 リアルタイムで笑い声を検出中（Ctrl+Cで終了）")
try:
    while True:
        audio = sd.rec(int(sr * duration), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        waveform = np.squeeze(audio)

        # YAMNetで分類
        scores, _, _ = model(waveform)
        mean_scores = np.mean(scores, axis=0)
        laughter_score = float(mean_scores[laughter_index])
        laughter_scores.append(laughter_score)

        # 表示
        print(f"[{time.strftime('%H:%M:%S')}] 😂 Laughter Score: {laughter_score:.3f}")

except KeyboardInterrupt:
    print("\n🛑 検出終了。グラフを描画します。")
    plt.plot(laughter_scores, marker='o')
    plt.title("リアルタイム Laughter スコア")
    plt.xlabel("秒数")
    plt.ylabel("Laughter Score")
    plt.grid(True)
    plt.show()
