# ------------------------------------------------------------------------------------
# Библиотеки и модули
# ------------------------------------------------------------------------------------
import numpy as np              # Библиотека NumPy для математичсеких операций и анализа данных
import os, warnings             # Библиотека работы с ОС и предупреждениями

import tensorflow as tf         # Библиотека Tensorflow с большим обьемом инстурментов машинного обучения
from tensorflow import keras    # Библиотека Keras для создания и работы с нейронными сетями
from tensorflow.keras.preprocessing.image import img_to_array

# Библиотека Tensorflow Hub для загрузки предобученных нейронных сетей
import tensorflow_hub as hub

import argparse # Библиотека ArgParse для обработки аргументов командной строки
                # и создания удобных командных интерфейсов

import imutils  # Библиотека ImUtils набор функций для загрузки, сохранения
                # и обработки изображений и видео
from imutils.video import VideoStream
from imutils.video import FPS

import time     # Библиотека Time c функциями работы с системным,
                # календарным и другими представлениями времени

import cv2      # Библиотека OpenCV для анализа и обработки изображений

# ------------------------------------------------------------------------------------
# Главная программа
# ------------------------------------------------------------------------------------

# Загрузка обученной раннее нейронной сети
print("[INFO] loading network...")
model = keras.models.load_model('king_crab_classification_model.keras', custom_objects={'KerasLayer': hub.KerasLayer})

# Запуск видео-потока
print("[INFO] starting video stream...")
vs = VideoStream( src=0).start()
time.sleep(2.0)
fps = FPS().start()
frame = vs.read()

# Начало основного цикла
while True:
    # Чтение кадра
    frame = vs.read()

    # Получение части изображения в размере 640*640
    # Размерности обусловлены видео-потоком с камеры
    # c разрешением 640 в ширину и 480 в высоту
    (h, w) = frame.shape[:2]
    frame_left_border = int((w-480)/2)
    frame_right_border = int((w+480)/2)
    frame = frame[:, frame_left_border:frame_right_border]
    cv2.imshow('Cropped image', imutils.resize(frame, width=400))

    # Подготовительная обработка изображения перед классификацией
    frame = imutils.resize(frame, width=640)
    image = frame.astype('float') / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Классифицирование изображения
    res = model.predict(image)[0][0]

    # Присваивание названия классу
    label = 'Other'
    prob = res
    if res < 0.5:
        prob = 1.0 - res
        label = 'King crab'

    print( 'Class =\t{},\tResult =\t{:.4f},\tConfidence =\t{:.4f}'.format(label, res, prob))
    label = '{}: {:.2f}%'.format(label, prob * 100)

    # Отрисовка результатов обработки на выходном изображении программы
    output = imutils.resize( frame, width=400)
    output = cv2.rectangle(output, (0, 0), (250, 35), (0, 0, 0), -1)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Отображение обработанного кадра
    cv2.imshow('Camera view', output)
    key = cv2.waitKey(1) & 0xFF

    # Нажатие клавиши 'q' - выход из цикла
    if key == ord("q"):
        break

    # Обновление FPS счетчика
    fps.update()

# Остановка таймера и отображение FPS информации
fps.stop()
# Завершение работы программы
cv2.destroyAllWindows()
vs.stop()