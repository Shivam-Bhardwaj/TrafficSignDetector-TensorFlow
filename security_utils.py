import os
import hashlib
import requests
import zipfile
from pathlib import Path
import joblib
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Dataset configuration
DATASET_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"  # Example URL, replace with actual
EXPECTED_HASH = "expected_sha256_hash_here"  # Replace with actual hash

class SecureDataLoader:
    """Secure data loader with validation."""

    def __init__(self, max_size_mb=1000):
        self.max_size_mb = max_size_mb

    def safe_load(self, file_path: Path):
        """Safely load data using joblib with validation."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.stat().st_size > self.max_size_mb * 1024 * 1024:
            raise ValueError("File too large")

        try:
            data = joblib.load(file_path)
            if not isinstance(data, dict) or 'features' not in data or 'labels' not in data:
                raise ValueError("Invalid data format")
            return data
        except Exception as e:
            raise ValueError(f"Failed to load data: {e}")

def secure_download_dataset(target_dir: str = "../dataset", url: str = DATASET_URL, expected_hash: str = EXPECTED_HASH):
    """Securely download and extract dataset with integrity check."""
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    zip_path = target_path / "dataset.zip"
    
    # Download with progress
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    if total_size > 1000 * 1024 * 1024:  # 1GB limit
        raise ValueError("Download too large")
    
    hasher = hashlib.sha256()
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            hasher.update(chunk)
    
    actual_hash = hasher.hexdigest()
    if actual_hash != expected_hash:
        zip_path.unlink()
        raise ValueError(f"Hash mismatch: expected {expected_hash}, got {actual_hash}")
    
    # Extract safely
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_path)
    
    zip_path.unlink()
    logger.info("Dataset downloaded and extracted securely")

def safe_load_traffic_data(data_dir: str):
    """Load traffic sign data securely."""
    loader = SecureDataLoader()
    data_path = Path(data_dir)
    
    train_data = loader.safe_load(data_path / "train.p")
    valid_data = loader.safe_load(data_path / "valid.p")
    test_data = loader.safe_load(data_path / "test.p")
    
    return train_data, valid_data, test_data
