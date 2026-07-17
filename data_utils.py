import os
import shutil
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import GENERATED_DIR, EXTRACT_DIR, IMG_SIZE, BATCH_SIZE

def get_file_list(input_dir):
    return [file for file in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, file))]

def split_dataset():
    emotions = ['surprise', 'sadness', 'anger', 'contempt', 'happy', 'disgust', 'fear']
    train_dir = os.path.join(GENERATED_DIR, 'train')
    validation_dir = os.path.join(GENERATED_DIR, 'validation')
    test_dir = os.path.join(GENERATED_DIR, 'test')
    
    for d in [train_dir, validation_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
        for emotion in emotions:
            os.makedirs(os.path.join(d, emotion), exist_ok=True)
            
    for emotion in emotions:
        original_dir = os.path.join(EXTRACT_DIR, 'ck', 'CK+48', emotion)
        if not os.path.exists(original_dir):
            print(f"Warning: {original_dir} not found. Skipping {emotion}.")
            continue
            
        train_dst = os.path.join(train_dir, emotion)
        val_dst = os.path.join(validation_dir, emotion)
        test_dst = os.path.join(test_dir, emotion)
        
        files = get_file_list(original_dir)
        total = len(files)
        
        # 70% Train
        train_files = random.sample(files, int(0.7 * total))
        for f in train_files: shutil.move(os.path.join(original_dir, f), train_dst)
        print(f'{emotion} train Done!')
        
        # 50% of remaining for Validation (approx 15% total)
        remaining = get_file_list(original_dir)
        val_files = random.sample(remaining, int(0.5 * len(remaining)))
        for f in val_files: shutil.move(os.path.join(original_dir, f), val_dst)
        print(f'{emotion} validation Done!')
        
        # Rest for Test
        for f in get_file_list(original_dir): shutil.move(os.path.join(original_dir, f), test_dst)
        print(f'{emotion} test Done!\n---------------')

def get_class_weights():
    train_dir = os.path.join(GENERATED_DIR, 'train')
    emotions = ['surprise', 'sadness', 'happy', 'fear', 'disgust', 'contempt', 'anger']
    label_list = [len(get_file_list(os.path.join(train_dir, e))) for e in emotions]
    max_val = max(label_list)
    return {i: float(max_val / count) if count > 0 else 1.0 for i, count in enumerate(label_list)}

def get_generators():
    train_dir = os.path.join(GENERATED_DIR, 'train')
    validation_dir = os.path.join(GENERATED_DIR, 'validation')
    test_dir = os.path.join(GENERATED_DIR, 'test')
    
    train_datagen = ImageDataGenerator(
        preprocessing_function=lambda x: (x / 127.5) - 1.0,
        rotation_range=40, width_shift_range=0.25, height_shift_range=0.25,
        shear_range=0.25, zoom_range=0.25, horizontal_flip=True, fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(preprocessing_function=lambda x: (x / 127.5) - 1.0)
    
    train_gen = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
    val_gen = test_datagen.flow_from_directory(validation_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
    test_gen = test_datagen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=1, shuffle=False, class_mode='categorical')
        
    return train_gen, val_gen, test_gen