import os

# Base directory for saving models and datasets
BASE_DIR = './project_data' 
os.makedirs(BASE_DIR, exist_ok=True)

# Dataset paths
# Note: The Kaggle signed URL may expire. If it fails, download CK+ manually and place it in EXTRACT_DIR.
DATASET_URL = "https://storage.googleapis.com/kaggle-data-sets/65125/128470/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20230713%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230713T184454Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=532508f50062fa30a107a155ccf6a10d784f05e1ea9a0adee2559ecfbed8878b6793635e3495822bd7cbec5ef57f3eb5fae2f6696d511bfb2ae50bf06fe1db7a8fadd4a3ede2591e2f0cb6251f19dbcaa8cd01708f25c596960d2a2c16602bfd5f963cf01766fc436796b21efb121e9deb69ca2075e5d75b9e28a151f3a990b8c83d0faf4f27e4c248ebed040e11c1b9b9445ac2991feaef4b6a26433ed29858ab50abcb8c65c58abb0428358c023f8e3c01f81fcac6ac8579326872df2e2f213e93f81a89475169867e38a672c14f62b06c72b4e606dfc23c50b0d42807b4cf7abe9e77039ea8fa48165cf2d890dcca666b07e2436ba7a13e8d8fee2a2f6e25"
EXTRACT_DIR = os.path.join(BASE_DIR, 'CKplus_48.28')
GENERATED_DIR = os.path.join(BASE_DIR, 'generatedck+')

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

BASE_MODEL_PATH = os.path.join(MODEL_DIR, 'ENV2B2CK+.h5')
QAT_H5_PATH = os.path.join(MODEL_DIR, 'ENV2B2quantized.h5')
QAT_TFLITE_PATH = os.path.join(MODEL_DIR, 'ENV2B2QAT.tflite')
PTQ_TFLITE_PATH = os.path.join(MODEL_DIR, 'ENV2B2PTQ.tflite')

# Image parameters
IMG_SIZE = (260, 260)
BATCH_SIZE = 16
NUM_CLASSES = 7