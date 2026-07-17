import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import EfficientNetV2B2
from tensorflow.keras.callbacks import Callback
from config import BASE_MODEL_PATH, IMG_SIZE, NUM_CLASSES
from data_utils import get_class_weights, get_generators

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') > 0.98 and logs.get('accuracy') > 0.98:
            print("\nReached 98% val_accuracy so cancelling training!")
            self.model.stop_training = True
        else:
            print(" .It's Not good yet! ")

if __name__ == "__main__":
    train_gen, val_gen, _ = get_generators()
    class_weight = get_class_weights()
    
    pre_trained = EfficientNetV2B2(include_top=True, include_preprocessing=False, pooling=None,
                                   weights='imagenet', input_tensor=Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))
    x = layers.Dense(NUM_CLASSES, activation='softmax')(pre_trained.get_layer('top_dropout').output)
    model = models.Model(pre_trained.input, x)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_gen, steps_per_epoch=42, epochs=60, validation_data=val_gen, 
              validation_steps=9, class_weight=class_weight, callbacks=[MyCallback()])
    
    model.save(BASE_MODEL_PATH)
    print(f"Base model saved to {BASE_MODEL_PATH}")