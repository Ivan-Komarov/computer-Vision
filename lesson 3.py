import cv2
import numpy as np
from numpy.ma.core import filled

#створюємо матрицю із 0. 512х512 з трьома шарами кольорів
img = np.zeros((512, 512, 3), np.uint8)

# зафарбуємо іншим кольором фігуру для цього нам потрібно зробити зрізи наступним чином:
# img[:] = 109, 238, 135
#RGB = BGR
# : - значення всіх елементів матриці

# якщо робимо відступи, а не все зображення заливаємо кольором
# перша У потім Х
# img[100:150, 200:280] = 109, 238, 135



# створення прямокутника через вбудовані функції
cv2.rectangle(img, (100, 100), (200, 200), (109, 238, 135), thickness = 2) #верхня ліва координата, права нижня, колір, товщина контуру, cv2.FILLED - залита

# створення лінії
cv2.line(img, (100, 100), (200, 200), (109, 238, 135), thickness = 2)

print(img.shape)
cv2.line(img, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2), (109, 238, 135), thickness = 2)

# створення кола
cv2.circle(img, (200, 200), 20, (109, 238, 135), thickness = 2)#center, radius, color, fill/contur

#створення напису
cv2.putText(img, "name", (200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
#object, text, place, fonts, font-size, color


cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()