# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 21:22:26 2024

@author: Alex
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
import tensorflowjs as tfjs

# 設置參數
image_size = (300, 300)
image_path = 'images'
batch_size = 32

# 準備數據集並進行數據增強
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)


train_generator = train_datagen.flow_from_directory(
    image_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    image_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 使用預訓練的 VGG16 模型
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(300, 300, 3))
base_model.trainable = False  # 冻结预训练模型的层


# 增加自定义的全连接层
x = base_model.output
x = Flatten(name='flatten')(x)
x = Dense(256, activation='relu', name='dense1')(x)
x = Dropout(0.5)(x)  # 增加 Dropout 层
output_layer = Dense(train_generator.num_classes, activation='softmax', name='output')(x)

# 定義模型
model = Model(inputs=base_model.input, outputs=output_layer)

# 編譯模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
model.fit(train_generator, epochs=50, validation_data=validation_generator)

model.summary()
# 保存模型為 HDF5 文件
# model.save('model.h5')

# 保存模型為 TensorFlow.js 格式
path = 'tfjs_model'
tfjs.converters.save_keras_model(model, path)

with open(f'{path}/labels.txt', 'w') as f:
    for name in train_generator.class_indices.keys():
        f.write(f'{name}\n')
