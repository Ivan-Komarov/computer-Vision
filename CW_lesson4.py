# Обробка фото та виявлення контурів у Python з OpenCV
import cv2
import numpy as np

#Завантаження фото
img = cv2.imread('images/woman.jpg')

#Зміна розміру
scale = 1
img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
print(img.shape)

#Копія для малювання контурів і тексту
img_copy = img.copy()

#в сірий та розмиття
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (9,9), 2)

#посилення контрасту
img = cv2.equalizeHist(img)

#виявлення країв
img_edges = cv2.Canny(img, 170, 170)  # більш чутливі пороги

#пошук контурів
contours, hierarchy = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# contours — список всіх контурів, які знайшов cv2.findContours().
# cnt — це один контур з цього списку. Контур у OpenCV — це масив точок (x, y),
# які описують форму об’єкта на зображенні.

for cnt in contours:
    # Обчислення площі контуру
    area = cv2.contourArea(cnt)
    # cv2.contourArea(cnt) — функція OpenCV, яка повертає
    # площу контуру в пікселях. Наприклад, якщо контур маленький(шум), площа
    # буде дуже маленька.
    if area > 100:  # фільтр шуму
        # Якщо площа контуру менше 100 пікселів, його вважаємо шумом і
        # пропускаємо. Це дозволяє не малювати маленькі зайві контури.
        x, y, w, h = cv2.boundingRect(cnt) # повертає найменший прямокутник, який повністю огортає контур.

        # Контур
        cv2.drawContours(img_copy, [cnt], -1, (0,255,0), 2)

# [cnt] — контур або список контурів, які хочемо намалювати.
# -1 — малювати всі контури з масиву, якщо це список контурів (тут один).



        # Прямокутник
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255,0,0), 2)

        text_y = y - 5 if y - 5 > 10 else y + 15
#Щоб текст не виходив за межі зображення, перевіряємо y-5.
# Якщо y-5 > 10, текст ставимо над прямокутником (y-5).
# Інакше — під прямокутником (y+15).
        text = f"x:{x}, y:{y}, S:{int(area)}"
        cv2.putText(img_copy, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)


cv2.imshow("Contours and Coordinates", img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

