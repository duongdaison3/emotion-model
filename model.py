import tensorflow as tf
from tensorflow.keras import layers
import config

def build_model(num_classes):
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_shape=config.IMG_SIZE + (3,)
    )
    base_model.trainable = False

    # --- SỬA LẠI PHẦN NÀY ---
    # Thay thế [...] bằng các lớp tăng cường dữ liệu thực tế
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1)
    ], name="data_augmentation")
    # --- KẾT THÚC PHẦN SỬA ---
    
    inputs = tf.keras.Input(shape=config.IMG_SIZE + (3,))
    # Áp dụng data augmentation sau khi rescaling
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x) # Chuẩn hóa pixel về [0, 1]
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    # Thêm các lớp Dense để giảm overfitting
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        metrics=['accuracy']
    )
    return model