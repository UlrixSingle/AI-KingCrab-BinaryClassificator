# ------------------------------------------------------------------------------------
# Библиотеки и модули
# ------------------------------------------------------------------------------------
import matplotlib.pyplot as plt # Библиотека Matplotlib для отрисовки графов
from matplotlib import gridspec
import numpy as np              # Библиотека NumPy для математичсеких операций 
                                # и анализа данных
import pandas as pd             # Библиотека Pandas для обработки и анализа данных
import os, warnings             # Библиотека работы с ОС и предупреждениями

import tensorflow as tf         # Библиотека Tensorflow с большим обьемом инструментов 
                                # машинного обучения
from tensorflow import keras    # Библиотека Keras для создания и работы 
                                # с нейронными сетями
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Библиотека Tensorflow Hub для загрузки предобученных нейронных сетей
import tensorflow_hub as hub

# ------------------------------------------------------------------------------------
# Подготовительные меры
# ------------------------------------------------------------------------------------

# Параметр воспроизводимости (установка случайной генерации чисел)
def set_seed( seed = 31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed()

# Установка Matplotlib по умолчанию
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # Для очистки выходных клеток

# ------------------------------------------------------------------------------------
# Загрузка учебной и проверочной выборок набора данных
# ------------------------------------------------------------------------------------
ds_train_ = image_dataset_from_directory(   # Учебная выборка
    'input/DataSet/train',
    labels='inferred',
    label_mode='binary',
    image_size=[640, 640],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)
ds_valid_ = image_dataset_from_directory(   # Проверочная выборка
    'input/DataSet/valid',
    labels='inferred',
    label_mode='binary',
    image_size=[640, 640],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)

# ------------------------------------------------------------------------------------
# Настройка потока данных
# ------------------------------------------------------------------------------------
def convert_to_float( image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
ds_valid = (
    ds_valid_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

# ------------------------------------------------------------------------------------
# Создание и обучение модели
# ------------------------------------------------------------------------------------
def create_model(): # Функция создания модели
    # Предобученная сверточная сеть в качестве основания
    pretrained_base = hub.KerasLayer(
        'inceptionv3',
        trainable = False)

    # Модель
    tmp_model = keras.Sequential([
        pretrained_base,
        layers.Flatten(),
        # Присоединение заглавной части модели
        layers.Dense( 128, activation = 'relu'),
        layers.Dropout( 0.3),
        layers.BatchNormalization(),
        layers.Dense( 128, activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense( 1, activation = 'sigmoid')
    ])

    # Компиляция модели
    optimizer = tf.keras.optimizers.Adam( epsilon=0.01)
    tmp_model.compile(
        optimizer='adam',
        loss = 'binary_crossentropy',
        metrics=['binary_accuracy'],
    )
    return tmp_model

# Методика ранней остановки
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,    # Минимальное изменение в значении потерь считающееся за улучшение работы модели
    patience=10,        # Количество эпох, которые будут ожидаться перед остановкой обучения
    restore_best_weights=True   # Возвращение весов модели при ее лучшей производительности
)

# Обучение модели
model = create_model()
history = model.fit(
    ds_train,
    validation_data=ds_valid,
    callbacks=[early_stopping],
    epochs=100
)

# ------------------------------------------------------------------------------------
# Результаты обучения модели
# ------------------------------------------------------------------------------------

# Сохранение лучшей версии модели
model.save( 'kingcrab_classification_model.keras')

# Отрисовка истории обучения
history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
plt.show()