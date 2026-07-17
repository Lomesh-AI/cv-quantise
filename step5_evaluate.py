import os
import time
import numpy as np
import tensorflow as tf
from config import BASE_MODEL_PATH, QAT_TFLITE_PATH, PTQ_TFLITE_PATH
from data_utils import get_generators

def evaluate_tflite(model_path, test_gen, num_images):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    in_det, out_det = interpreter.get_input_details(), interpreter.get_output_details()
    correct, total, t_time = 0, 0, 0.0

    for x, y_true in test_gen:
        if total >= num_images: break
        if x.shape != tuple(in_det[0]['shape']): x = np.reshape(x, in_det[0]['shape'])
        x = x.astype(in_det[0]['dtype'])
        interpreter.set_tensor(in_det[0]['index'], x)
        
        start = time.time()
        interpreter.invoke()
        t_time += time.time() - start
        
        y_pred = np.argmax(interpreter.get_tensor(out_det[0]['index']), axis=-1)
        correct += np.sum(y_pred == np.argmax(y_true, axis=-1))
        total += x.shape[0]
    return correct / total, t_time / total

if __name__ == "__main__":
    _, _, test_gen = get_generators()
    
    print(f'Simple model size: {os.path.getsize(BASE_MODEL_PATH)} bytes')
    print(f'QAT model size: {os.path.getsize(QAT_TFLITE_PATH)} bytes')
    print(f'PTQ model size: {os.path.getsize(PTQ_TFLITE_PATH)} bytes\n')
    
    acc, t = evaluate_tflite(QAT_TFLITE_PATH, test_gen, 152)
    print(f"QAT TFLite -> Accuracy: {acc:.4f}, Avg Inference Time: {t:.5f}s")
    
    acc, t = evaluate_tflite(PTQ_TFLITE_PATH, test_gen, 152)
    print(f"PTQ TFLite -> Accuracy: {acc:.4f}, Avg Inference Time: {t:.5f}s\n")
    
    # Base model inference time
    model = tf.keras.models.load_model(BASE_MODEL_PATH)
    start = time.time()
    model.predict(test_gen)
    t_total = time.time() - start
    print(f"Base Model -> Total inference time: {t_total:.2f}s, Avg per image: {t_total/len(test_gen):.5f}s")