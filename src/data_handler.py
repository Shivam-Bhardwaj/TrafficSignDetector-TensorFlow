#!/usr/bin/env python3
"""
Data handling utilities for the Traffic Sign Classifier project.
This module provides functions for downloading, extracting, loading, and preprocessing the dataset.
"""

import os
import hashlib
import zipfile
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import joblib
import numpy as np
import logging
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureDataLoader:
    """Secure data loading with validation and integrity checks."""
    
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB max download
    ALLOWED_EXTENSIONS = {'.p', '.pkl', '.h5', '.hdf5', '.npz', '.npy', '.zip'}
    
    def __init__(self, base_dir: str = "dataset"):
        """Initialize secure data loader.
        
        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_url(self, url: str) -> bool:
        """Validate URL for security."""
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ('https', 'http'):
            logger.error(f"Invalid URL scheme: {parsed.scheme}")
            return False
        return True
    
    def calculate_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file hash for integrity verification."""
        hash_obj = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def secure_download(self, url: str, filename: str, 
                       expected_hash: Optional[str] = None) -> Path:
        """Securely download and verify file."""
        if not self.validate_url(url):
            raise SecurityError(f"URL failed security validation: {url}")
        
        file_path = self.base_dir / filename
        
        if file_path.exists() and expected_hash:
            if self.calculate_file_hash(file_path) == expected_hash:
                logger.info(f"File {filename} already exists and is valid")
                return file_path
        
        logger.info(f"Downloading {url} to {file_path}")
        
        with urllib.request.urlopen(url) as response:
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.MAX_FILE_SIZE:
                raise SecurityError(f"File too large: {content_length} bytes")
            
            with open(file_path, 'wb') as f:
                downloaded = 0
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    if downloaded > self.MAX_FILE_SIZE:
                        file_path.unlink()
                        raise SecurityError("Downloaded file exceeds size limit")
                    f.write(chunk)
        
        if expected_hash:
            actual_hash = self.calculate_file_hash(file_path)
            if actual_hash != expected_hash:
                file_path.unlink()
                raise SecurityError(f"Hash mismatch. Expected: {expected_hash}, Got: {actual_hash}")
        
        logger.info(f"Successfully downloaded and verified {filename}")
        return file_path
    
    def safe_extract_zip(self, zip_path: Path, extract_to: Optional[Path] = None) -> Path:
        """Safely extract ZIP file with path traversal protection."""
        if extract_to is None:
            extract_to = self.base_dir
        
        extract_to = Path(extract_to).resolve()
        
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            for member in zip_file.namelist():
                if os.path.isabs(member) or ".." in member:
                    raise SecurityError(f"Unsafe path in ZIP: {member}")
                zip_file.extract(member, extract_to)
        
        logger.info(f"Safely extracted {zip_path} to {extract_to}")
        return extract_to
    
    def safe_load_pickle(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Safely load pickle files using joblib."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.suffix not in self.ALLOWED_EXTENSIONS:
            raise SecurityError(f"File type not allowed: {file_path.suffix}")
        if file_path.stat().st_size > self.MAX_FILE_SIZE:
            raise SecurityError(f"File too large: {file_path}")
        
        try:
            data = joblib.load(file_path)
            logger.info(f"Successfully loaded data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise

def download_and_extract_dataset(data_dir: str = "dataset"):
    """Download and verify the traffic signs dataset securely."""
    loader = SecureDataLoader(base_dir=data_dir)
    url = 'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip'
    filename = 'traffic-signs-data.zip'
    zip_path = loader.secure_download(url, filename)
    loader.safe_extract_zip(zip_path)
    zip_path.unlink()
    logger.info("Dataset downloaded and extracted successfully.")

def load_traffic_data(data_dir: str = "dataset") -> Tuple[Dict, Dict, Dict]:
    """Safely load traffic sign data with validation."""
    loader = SecureDataLoader(data_dir)
    train_data = loader.safe_load_pickle(Path(data_dir) / 'train.p')
    valid_data = loader.safe_load_pickle(Path(data_dir) / 'valid.p')
    test_data = loader.safe_load_pickle(Path(data_dir) / 'test.p')
    return train_data, valid_data, test_data

def preprocess_images(X: np.ndarray) -> np.ndarray:
    """
    Preprocess images by converting to grayscale and normalizing.
    """
    # Convert to grayscale
    X_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in X])
    
    # Add channel dimension
    X_gray = X_gray[..., np.newaxis]
    
    # Normalize
    X_normalized = (X_gray - 128) / 128
    
    return X_normalized

class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass
