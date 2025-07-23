# data_loader.py (Code đúng)
import tensorflow as tf
import config

def load_datasets():
    # Bước 1: Tải cả 3 bộ dữ liệu
    train_data = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        label_mode='categorical',
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE
    )
    valid_data = tf.keras.utils.image_dataset_from_directory(
        config.VALID_DIR,
        label_mode='categorical',
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE
    )
    test_data = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        label_mode='categorical',
        image_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )

    # Bước 2: Lấy class_names ra TRƯỚC KHI tối ưu
    class_names = train_data.class_names

    # Bước 3: Tối ưu pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
    valid_data = valid_data.cache().prefetch(buffer_size=AUTOTUNE)
    test_data = test_data.cache().prefetch(buffer_size=AUTOTUNE)

    # Bước 4: Trả về các biến đã xử lý
    return train_data, valid_data, test_data, class_names