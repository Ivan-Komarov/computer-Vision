
import tensorflow as tf

# layers — набори готових шарів (Convolution, Pooling, Dense...)
# models — дозволяє створювати архітектуру мережі
from tensorflow.keras import layers, models


import numpy as np

# image — завантаження окремих зображень для перевірки моделі
from tensorflow.keras.preprocessing import image


#завантаження файлів
# Цей метод заходить у папку data/train,
# автоматично читає назви папок (cars, cats, dogs),
# і кожному файлу всередині присвоює відповідну мітку класу.
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/train",            # шлях до тренувальних зображень
    image_size=(128, 128),      # масштабування кожного зображення
    batch_size=32,              # кількість фото за один крок навчання
    label_mode="categorical"    # повертає мітку у форматі [1,0,0]
)

# Те саме для тестового набору
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/test",
    image_size=(128, 128),
    batch_size=32,
    label_mode="categorical"
)



# нормалізація зображень

# Реальні пікселі 0–255. Мережам зручніше працювати з 0–1.
normalization_layer = layers.Rescaling(1./255)

# map застосовує нормалізацію до кожного батчу
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds  = test_ds.map(lambda x, y: (normalization_layer(x), y))


#побудова моделі

# Створюємо послідовну модель: шари додаються один за одним
model = models.Sequential()


# виділення простих ознак (краї, лінії)
model.add(layers.Conv2D(
    filters=32,                # кількість фільтрів
    kernel_size=(3, 3),        # розмір фільтра
    activation='relu',         # функція активації
    input_shape=(128, 128, 3)  # форма вхідного зображення (RGB)
))
model.add(layers.MaxPooling2D((2, 2)))   # зменшуємо карту ознак у 2 рази


#глибші ознаки (контури, структури)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))


# найскладніші ознаки (форми об'єктів)
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))


# перетворення 3D → 1D
# Нам потрібно передати результат у Dense-шари,
# тому перетворюємо карти ознак у вектор.
model.add(layers.Flatten())


# фінальні шари класифікації
model.add(layers.Dense(64, activation='relu'))   # внутрішній шар
model.add(layers.Dense(3, activation='softmax')) # 3 класи: cats, dogs, cars


#компіляція моделі

# optimizer — алгоритм корекції ваг
# loss — функція, яка вимірює "помилку"
# metrics — що хочемо бачити в результаті
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


#навчання моделі

# epochs — кількість циклів проходу по всьому train набору
history = model.fit(
    train_ds,
    epochs=10,
    validation_data=test_ds   # оцінка точності після кожної епохи
)


#оцінка якості

test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)


#перевірка

# Назви класів — повинні відповідати назвам папок
class_names = ["cars", "cats", "dogs"]

# Завантажуємо зображення для класифікації
img = image.load_img("images/woman.jpg", target_size=(128, 128))

# Перетворення у масив 128×128×3
img_array = image.img_to_array(img)

# Нормалізація (0–1)
img_array = img_array / 255.0

# Додаємо batch-вимір: тепер форма (1,128,128,3)
img_array = np.expand_dims(img_array, axis=0)

# Модель робить прогноз
predictions = model.predict(img_array)

# argmax повертає індекс класу з найбільшою ймовірністю
predicted_index = np.argmax(predictions[0])

# Виводимо результат
print("Ймовірності по класах:", predictions[0])
print("Модель визначила:", class_names[predicted_index])