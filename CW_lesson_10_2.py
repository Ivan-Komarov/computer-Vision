import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# навчання по кольорам

colors = {
    "red": (0, 0, 255),
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255)
}

X = []
y = []

for name, bgr in colors.items():
    for i in range(20):
        noise = np.random.randint(-20, 20, 3)
        sample = np.clip(np.array(bgr) + noise, 0, 255)
        X.append(sample)
        y.append(name)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
print("✅ Модель кольорів навчена.")


# відео і відстеження

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # створюємо маску для яскравих кольорів
    mask = cv2.inRange(hsv, (20, 50, 50), (255, 255, 255)) #багато білих плям, шум

    # очищення маски (морфологія)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))#прибрали дрібні точки
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))#закрили дірки в середині об’єктів

    # знаходимо контури об’єктів
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # ігноруємо маленькі плями
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y:y+h, x:x+w]

            # обчислюємо середній колір об’єкта
            mean_color = cv2.mean(roi)[:3]
            mean_color = np.array(mean_color).reshape(1, -1)

            # передбачення кольору
            label = model.predict(mean_color)[0]

            # рамка і підпис
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
            cv2.putText(frame, label.upper(), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # результат
    cv2.imshow("Color", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
