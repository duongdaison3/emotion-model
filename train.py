# train.py
import tensorflow as tf
import config
from data_loader import load_datasets
from model import build_model

if __name__ == '__main__':
    # 1. Tải dữ liệu
    train_ds, valid_ds, test_ds, class_names = load_datasets()
    print("Các lớp cảm xúc:", class_names)
    
    # 2. Xây dựng model
    num_classes = len(class_names)
    emotion_model = build_model(num_classes)
    emotion_model.summary()
    
    # 3. Huấn luyện model
    print("Bắt đầu huấn luyện...")
    history = emotion_model.fit(
        train_ds,
        epochs=config.EPOCHS,
        validation_data=valid_ds
    )
    
    # 4. Lưu model
    emotion_model.save(config.MODEL_SAVE_PATH)
    print(f"Model đã được lưu tại: {config.MODEL_SAVE_PATH}")
    
    # (Tùy chọn) Có thể gọi hàm đánh giá và vẽ biểu đồ ở đây