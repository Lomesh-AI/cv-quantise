import tensorflow as tf
from config import BASE_MODEL_PATH, PTQ_TFLITE_PATH

if __name__ == "__main__":
    model = tf.keras.models.load_model(BASE_MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open(PTQ_TFLITE_PATH, 'wb') as f: f.write(tflite_model)
    print(f"PTQ model saved to {PTQ_TFLITE_PATH}")