#Маски та аналіз контурів об’єктів в OpenCV
import cv2
import numpy as np

# Завантаження зображення
img = cv2.imread('images/woman.jpg')
img_copy = img.copy()

# Переведення в HSV відтінок, насиченість, яскравість
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower = np.array([0, 7, 0])
upper = np.array([179, 255, 255])
mask = cv2.inRange(img, lower, upper)

# Щоб показати тільки виділений об’єкт у кольорі
img = cv2.bitwise_and(img, img, mask=mask) #вхідне зображення, зображення з яким застосовується накладання

#Частина 2. Аналіз контурів
# Коли маску вже створено, можна знайти контури — межі об’єктів.
# Контури — це точки, що утворюють лінію навколо предмета.
# Вони допомагають обчислити розмір, форму, центр і пропорції

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# RETR_EXTERNAL — знаходить лише зовнішні контури.
# CHAIN_APPROX_SIMPLE — зберігає тільки ключові точки контуру, що економить пам’ять.
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 200:  # ігноруємо маленькі шуми
        perimeter = cv2.arcLength(cnt, True)#Обчислюється периметр (довжина дуги) контуру. Другий аргумент True вказує, що контур є замкненим.
        M = cv2.moments(cnt)#Обчислюються моменти контуру. Це математичні характеристики, які описують форму, розмір та орієнтацію об'єкта.

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # (Центр мас): З моментів розраховується центр
            # мас(або центроїд) об'єкта. Це "середня" позиція контуру. Перевірка M["m00"] != 0 запобігає діленню на нуль, якщо контур порожній.


        x, y, w, h = cv2.boundingRect(cnt) #прямокутний обмежувальний контур
        aspect_ratio = round(w / h, 2) #Обчислюється як ширина/висота (w/h). Це допомагає відрізняти, наприклад, довгий прямокутник від квадрата.
        compactness = round((4 * np.pi * area) / (perimeter ** 2), 2) #Обчислюється як міра округлості об'єкта. Значення близьке до 1.0 вказує на те,
        # що форма близька до ідеального кола (це співвідношення площі кола до площі з тим самим периметром).

        #  визначаємо форму
        #ключова функція, яка апроксимує контур до форми з меншою кількістю вершин.
        # Вона "випрямляє" криві та ігнорує дрібні нерівності, залишаючи лише основні кути.
        # 0.02 * perimeter — це точність апроксимації. Чим більше це значення, тим менше вершин буде в результаті.
        # Визначення за кількістю вершин:
        # Якщо в апроксимованій формі 3 вершини (len(approx) == 3), це вважається Трикутником.
        # Якщо 4 вершини, це Чотирикутник (квадрат, прямокутник, ромб тощо).
        # Якщо більше 8 вершин (len(approx) > 8), це означає, що форма дуже гладка і має багато точок,
        # тому вона класифікується як Коло (або еліпс).
        # Інакше — "Інша форма".

        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 3:
            shape = "Trikutnik"
        elif len(approx) == 4:
            shape = "4jtirikutik"
        elif len(approx) > 8:
            shape = "oval"
        else:
            shape = "inshe"


        # Малюємо результати
        cv2.drawContours(img, [cnt], -1, (255,255,255), 2)
        cv2.circle(img_copy, (cx, cy), 4, (255,0,0), -1)
        cv2.putText(img_copy, f"{shape}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img_copy, f"A:{int(area)} P:{int(perimeter)}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.putText(img_copy, f"AR:{aspect_ratio} C:{compactness}", (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)

        text_y = y - 5 if y - 5 > 10 else y + 15
        # Щоб текст не виходив за межі зображення, перевіряємо y-5.
        # Якщо y-5 > 10, текст ставимо над прямокутником (y-5).
        # Інакше — під прямокутником (y+15).
        text = f"x:{x}, y:{y}, S:{int(area)}"
        cv2.putText(img_copy, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


cv2.imshow('Mask', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
