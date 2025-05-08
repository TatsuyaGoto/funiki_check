import cv2
from fer import FER
import time
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

detector = FER(mtcnn=False)
cap = cv2.VideoCapture(0)

interval = 1
graph_interval = 10
plot_duration = 300  # 5分
frame_interval = 1

happy_counts = deque(maxlen=plot_duration // graph_interval)
timestamps = deque(maxlen=plot_duration // graph_interval)

temp_happy_count = 0
temp_sample_count = 0
last_eval_time = time.time()
last_graph_time = time.time()

# matplotlib描画設定（バックエンドはAgg）
fig, ax = plt.subplots(figsize=(4, 3))
canvas = FigureCanvas(fig)

def plot_to_image():
    ax.clear()
    ax.set_title("Avg Happy (10s)")
    ax.set_ylim(0, 5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Avg Happy Count")
    ax.plot(happy_counts, '-o')

    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf, dtype=np.uint8)[..., :3]  # RGBに変換
    return img


while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    if now - last_eval_time > frame_interval:
        results = detector.detect_emotions(frame)
        happy_this_frame = sum(
            1 for person in results if max(person["emotions"], key=person["emotions"].get) == "happy"
        )
        temp_happy_count += happy_this_frame
        temp_sample_count += 1
        last_eval_time = now

    if now - last_graph_time > graph_interval:
        avg_happy = temp_happy_count / temp_sample_count if temp_sample_count else 0
        happy_counts.append(avg_happy)
        timestamps.append(time.strftime('%H:%M:%S'))

        temp_happy_count = 0
        temp_sample_count = 0
        last_graph_time = now

    for person in results:
        (x, y, w, h) = person["box"]
        emotions = person["emotions"]
        dominant = max(emotions, key=emotions.get)
        score = emotions[dominant] * 100
        label = f"{dominant} ({score:.1f}%)"
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # グラフ画像生成＆リサイズ
    graph_img = plot_to_image()
    graph_img = cv2.resize(graph_img, (frame.shape[1], frame.shape[0]))

    # 横に連結して表示
    combined = np.hstack((frame, graph_img))
    cv2.imshow("Emotion + Graph", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
