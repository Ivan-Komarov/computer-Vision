# 1
# основні бібліотеки для обробки даних і побудови нейронної мережі
import pandas as pd              # робота з таблицями CSV
import numpy as np               # математичні операції
import tensorflow as tf          # бібліотека для нейронних мереж (Google)
from tensorflow import keras     # високорівневий API для TensorFlow
from tensorflow.keras import layers   # використовується для створення шарів
from sklearn.preprocessing import LabelEncoder  # перетворення текстових міток у числа
import matplotlib.pyplot as plt  # побудова графіків

# 2
# CSVмістить: площу, периметр, кількість кутів і тип фігури
df = pd.read_csv('data/figures.csv')   # зчитуємо файл

print(df.head())                  # виводимо перші 5 рядків для перевірки

# 3
# перетворюємо назви фігур ('triangle', 'square', 'circle') у числа (0,1,2)
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

# вибираємо стовпці для навчання (вхідні ознаки)
X = df[['area', 'perimeter', 'corners']]  # Вхід (матриця ознак)
y = df['label_enc']                        # Вихід (мітки класів)

# 4 створення моделі
# модель типу "Sequential" означає, що шари розташовані послідовно один за одним
model = keras.Sequential([
    # перший шар: 8 нейронів, функція активації ReLU
    # input_shape=(3,) означає, що на вхід подається 3 параметри (area, perimeter, corners)
    layers.Dense(8, activation='relu', input_shape=(3,)),

    # другий прихований шар — теж 8 нейронів, щоб мережа краще "запам’ятала" закономірності
    layers.Dense(8, activation='relu'),

    # вихідний шар: 3 нейрони (по одному на кожен клас)
    # використовуємо softmax, щоб отримати ймовірності кожного класу
    layers.Dense(3, activation='softmax')
])

# 5 компіляція моделі
# на цьому етапі ми визначаємо, як мережа буде навчатись:
model.compile(
    optimizer='adam',                        # Алгоритм, який підбирає найкращі ваги
    loss='sparse_categorical_crossentropy',  # Функція втрат для багатокласової класифікації
    metrics=['accuracy']                     # Ми хочемо бачити точність у відсотках
)

# 6 навчання
# навчання (метод fit) проходить у кілька "епох".
# epoch = один повний прохід усіх даних через мережу.
# verbose=0 — вимикає детальний вивід, щоб не перевантажувати консоль.
history = model.fit(X, y, epochs=200, verbose=0)

# 7 візуалізація навчання
# під час навчання ми можемо спостерігати, як змінюється помилка (loss)
# і точність (accuracy) від епохи до епохи.
plt.plot(history.history['loss'], label='Loss (втрата)')
plt.plot(history.history['accuracy'], label='Accuracy (точність)')
plt.xlabel('Epoch (епоха)')
plt.ylabel('Значення')
plt.title('Процес навчання моделі')
plt.legend()
plt.show()

# 8 тестування

test = np.array([[16, 16, 0]])

# отримуємо ймовірності кожного класу
pred = model.predict(test)


print("\nЙмовірності для кожного класу:", pred)
print("Модель визначила:", encoder.inverse_transform([np.argmax(pred)]))

# np.argmax(pred) — знаходить індекс найбільшої ймовірності
# encoder.inverse_transform — перетворює число назад у назву фігури