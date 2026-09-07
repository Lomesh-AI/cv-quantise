import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.callbacks import LearningRateScheduler, Callback
from config import BASE_MODEL_PATH, QAT_H5_PATH, QAT_TFLITE_PATH
from data_utils import get_generators

def apply_quantization_layers(layer):
    if 'se_excite' in layer.name: return layer
    return tfmot.quantization.keras.quantize_annotate_layer(layer)

class StopOnValidationAcc(Callback):
    def __init__(self, target_val_acc=0.98):
        super().__init__()
        self.target_val_acc = target_val_acc
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') >= self.target_val_acc:
            print(f'\nReached {100 * self.target_val_acc}% validation accuracy, stopping training.')
            self.model.stop_training = True

def lr_schedule(epoch):
    lr = 1e-4
    if epoch >= 6: lr *= 0.1
    elif epoch >= 7: lr *= 0.01 # Note: Reproducing original notebook logic
    return lr

if __name__ == "__main__":
    train_gen, val_gen, _ = get_generators()
    
    # 1. Create and save initial quantized model
    with tfmot.quantization.keras.quantize_scope():
        base_model = tf.keras.models.load_model(BASE_MODEL_PATH)
    annotated = tf.keras.models.clone_model(base_model, clone_function=apply_quantization_layers)
    quant_model = tfmot.quantization.keras.quantize_apply(annotated)
    quant_model.save(QAT_H5_PATH)
    
    # 2. Load, compile, and fine-tune
    with tfmot.quantization.keras.quantize_scope():
        modelq = tf.keras.models.load_model(QAT_H5_PATH)
    modelq.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    modelq.fit(train_gen, steps_per_epoch=len(train_gen), epochs=10, validation_data=val_gen, 
               validation_steps=len(val_gen), callbacks=[LearningRateScheduler(lr_schedule), StopOnValidationAcc()])
    
    # 3. Convert fine-tuned model to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(modelq)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(QAT_TFLITE_PATH, 'wb') as f: f.write(converter.convert())
    print(f"QAT model saved to {QAT_H5_PATH} and {QAT_TFLITE_PATH}")