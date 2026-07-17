import os
import urllib
import zipfile
from config import EXTRACT_DIR, DATASET_URL
from data_utils import split_dataset

def download_data():
    if os.path.exists(os.path.join(EXTRACT_DIR, 'ck')):
        print("Dataset already extracted. Skipping download.")
        return
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    print("Attempting download... (If this fails due to expired URL, download CK+ manually and extract to EXTRACT_DIR)")
    try:
        zip_path, _ = urllib.request.urlretrieve(DATASET_URL)
        with zipfile.ZipFile(zip_path, "r") as f: f.extractall(EXTRACT_DIR)
        print("Download and extraction complete.")
    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    download_data()
    split_dataset()
    print("Data preparation complete.")