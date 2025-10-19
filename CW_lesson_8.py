import cv2
import numpy as np

# підвантаження моделі каскаду для облич
# face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_alt2.xml')
face_cascade = cv2.CascadeClassifier('data/lbpcascade_frontalface_improved.xml')

#2 додаємо очі в середині обличчя
eye_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')


#3 посмішка
smile_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_smile.xml")


#4 DNN модель
# face_net = cv2.dnn.readNetFromCaffe('data/DNN/deploy.prototxt','data/DNN/res10_300x300_ssd_iter_140000.caffemodel')




cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:# Якщо камера не відповідає — виходимо
        break

#     #__________________block DNN_______________
#     #frame — це кадр з вебкамери, який OpenCV представляє як масив NumPy.
#     #frame.shape повертає кортеж (висота, ширина, канали) для кольорового зображення.
#     # Наприклад: (480, 640, 3)
#     # frame.shape[:2] бере тільки перші два елементи (висота, ширина) і присвоює їх h та w.
#     (h, w) = frame.shape[:2]  # потрібно для масштабування координат обличчя
#
#
#
#     #blobFromImage перетворює зображення у формат, який розуміє DNN:
#     # масштабування: 1.0 — без зміни масштабів пікселів
#     # розмір: (300, 300) — DNN очікує квадрат 300×300 пікселів
#     # середнє віднімання: (104.0, 177.0, 123.0) — стандартні значення для передобробки,
#     # які видаляють середнє по каналах BGR
#     # Результат: багатовимірний масив для DNN, називається blob.
#     blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
#                                  (104.0, 177.0, 123.0))
#     face_net.setInput(blob)
#     detections = face_net.forward()
#     #setInput(blob) — передаємо оброблене зображення в нейронну мережу.
#     #forward() — виконує прямий прохід (forward pass) через мережу і повертає результати.
#     #detections — масив із знайденими обличчями та їхніми параметрами.
#
#     # Обробка кожного знайденого обличчя
#     for i in range(detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#     #detections.shape[2] — кількість виявлених об’єктів (облич).
#     #detections[0, 0, i, 2] — впевненість (confidence) мережі, наскільки це точно обличчя (0–1).
#
#         if confidence > 0.5:  # поріг впевненості
#             # Якщо мережа впевнена менше ніж на 50% — ми ігноруємо це обличчя.
#             # Це допомагає відкидати шуми і помилкові спрацьовування.
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (x, y, x2, y2) = box.astype("int")
#
#             # обмежуємо координати всередині кадру
#             x, y = max(0, x), max(0, y)
#             x2, y2 = min(w - 1, x2), min(h - 1, y2)
#
#             # малюємо прямокутник навколо обличчя
#             cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
#
#
#             # ROI для очей і посмішки
#             roi_gray = cv2.cvtColor(frame[y:y2, x:x2], cv2.COLOR_BGR2GRAY)
#             roi_color = frame[y:y2, x:x2]
#
#
#             # Виявлення очей всередині обличчя
#             eyes = eye_cascade.detectMultiScale(
#                 roi_gray,
#                 scaleFactor=1.1,
#                 minNeighbors=10,
#                 minSize=(15, 15)
#             )
#             for (ex, ey, ew, eh) in eyes:
#                 cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
#
#
#             # Виявлення посмішки всередині обличчя
#             smiles = smile_cascade.detectMultiScale(
#                 roi_gray,
#                 scaleFactor=1.7,
#                 minNeighbors=10,
#                 minSize=(25, 25)
#             )
#             for (sx, sy, sw, sh) in smiles:
#                 cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)
#
#
#     cv2.imshow('DNN Face Tracking', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#
# cap.release()
# cv2.destroyAllWindows()
#
#
# #___________________________________________________
    # Перетворення кадру у відтінки сірого для пришвидшення обчислень та відзеркалення
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.flip(gray, 1)

    # Виявлення облич на зображенні
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,  # коефіцієнт зменшення зображення при кожному проході(масштабування)
        minNeighbors = 5,  # кількість перевірок для підтвердження обличчя(кількість сусідів для фільтрації)
        minSize = (30, 30)  # мінімальний розмір області, яку вважаємо обличчям(мінімальний розмір обличчя)
    )
    #скорочено faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    # print(faces)

    # Малюємо прямокутники навколо знайдених облич
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


    #ROI для очей — область обличчя
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
    #це зріз частини зображення, де знаходиться обличчя.
    # Тобто ми беремо тільки ту область, де каскад побачив обличчя,
    # і передаємо її для подальшого пошуку очей.

    #виявлення очей у середині обличчя
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor = 1.1,minNeighbors = 10, minSize = (15, 15))
    # Малюємо прямокутники для очей
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)


    #все те саме, тільки для посмішки
            smiles = smile_cascade.detectMultiScale(
        roi_gray,
        scaleFactor=1.7,  # або 1.5–1.8
        minNeighbors=10,  # або 8–15
        minSize=(25, 25)  # мінімальний розмір, щоб уникнути шуму
    )

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)



    # Додаємо текст з кількістю знайдених облич
    cv2.putText(frame, f"Faces detected: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)


    cv2.imshow('Haar Face Tracking', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
