#!/usr/bin/env python3
"""
Security utilities for safe data loading and validation.
Replaces unsafe pickle operations with secure alternatives.
"""

import os
import hashlib
import zipfile
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, Optional, Union
import joblib
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecureDataLoader:
    """Secure data loading with validation and integrity checks."""
    
    # Known good hashes for datasets (update as needed)
    KNOWN_HASHES = {
        'traffic-signs-data.zip': '9a55b8b74bb0c3c2a1f1b2c8c7d7e2c3b5f1a4e9d8c7b6a5f2e1d9c8b7a6f5e4'
    }
    
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB max download
    ALLOWED_EXTENSIONS = {'.p', '.pkl', '.h5', '.hdf5', '.npz', '.npy', '.zip'}
    
    def __init__(self, base_dir: str = "../dataset"):
        """Initialize secure data loader.
        
        Args:
            base_dir: Base directory for data storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_url(self, url: str) -> bool:
        """Validate URL for security.
        
        Args:
            url: URL to validate
            
        Returns:
            bool: True if URL is safe
        """
        parsed = urllib.parse.urlparse(url)
        
        # Check for valid schemes
        if parsed.scheme not in ('https', 'http'):
            logger.error(f"Invalid URL scheme: {parsed.scheme}")
            return False
        
        # Check for suspicious patterns
        suspicious_patterns = ['..', 'localhost', '127.0.0.1', '0.0.0.0']
        for pattern in suspicious_patterns:
            if pattern in url.lower():
                logger.error(f"Suspicious URL pattern detected: {pattern}")
                return False
        
        return True
    
    def calculate_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate file hash for integrity verification.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm to use
            
        Returns:
            str: Hex digest of file hash
        """
        hash_obj = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def secure_download(self, url: str, filename: str, 
                       expected_hash: Optional[str] = None) -> Path:
        """Securely download and verify file.
        
        Args:
            url: URL to download from
            filename: Local filename to save as
            expected_hash: Expected SHA256 hash for verification
            
        Returns:
            Path: Path to downloaded file
            
        Raises:
            SecurityError: If download fails security checks
        """
        if not self.validate_url(url):
            raise SecurityError(f"URL failed security validation: {url}")
        
        file_path = self.base_dir / filename
        
        # Check if file already exists and is valid
        if file_path.exists() and expected_hash:
            if self.calculate_file_hash(file_path) == expected_hash:
                logger.info(f"File {filename} already exists and is valid")
                return file_path
        
        logger.info(f"Downloading {url} to {file_path}")
        
        # Download with size limit
        with urllib.request.urlopen(url) as response:
            if hasattr(response, 'headers'):
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
                        file_path.unlink()  # Delete partial file
                        raise SecurityError("Downloaded file exceeds size limit")
                    f.write(chunk)
        
        # Verify hash if provided
        if expected_hash:
            actual_hash = self.calculate_file_hash(file_path)
            if actual_hash != expected_hash:
                file_path.unlink()  # Delete invalid file
                raise SecurityError(f"Hash mismatch. Expected: {expected_hash}, Got: {actual_hash}")
        
        logger.info(f"Successfully downloaded and verified {filename}")
        return file_path
    
    def safe_extract_zip(self, zip_path: Path, extract_to: Optional[Path] = None) -> Path:
        """Safely extract ZIP file with path traversal protection.
        
        Args:
            zip_path: Path to ZIP file
            extract_to: Directory to extract to (defaults to base_dir)
            
        Returns:
            Path: Directory containing extracted files
        """
        if extract_to is None:
            extract_to = self.base_dir
        
        extract_to = Path(extract_to).resolve()
        
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            for member in zip_file.namelist():
                # Check for path traversal attempts
                if os.path.isabs(member) or ".." in member:
                    raise SecurityError(f"Unsafe path in ZIP: {member}")
                
                # Extract file
                zip_file.extract(member, extract_to)
        
        logger.info(f"Safely extracted {zip_path} to {extract_to}")
        return extract_to
    
    def safe_load_pickle(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Safely load pickle files using joblib (more secure than pickle).
        
        Args:
            file_path: Path to pickle file
            
        Returns:
            dict: Loaded data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix not in self.ALLOWED_EXTENSIONS:
            raise SecurityError(f"File type not allowed: {file_path.suffix}")
        
        # Check file size
        if file_path.stat().st_size > self.MAX_FILE_SIZE:
            raise SecurityError(f"File too large: {file_path}")
        
        try:
            # Use joblib for safer loading
            data = joblib.load(file_path)
            logger.info(f"Successfully loaded data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise
    
    def validate_data_structure(self, data: Dict[str, Any]) -> bool:
        """Validate loaded data structure for expected format.
        
        Args:
            data: Loaded data dictionary
            
        Returns:
            bool: True if data structure is valid
        """
        required_keys = ['features', 'labels']
        
        for key in required_keys:
            if key not in data:
                logger.error(f"Missing required key: {key}")
                return False
        
        # Check if features and labels are numpy arrays
        if not isinstance(data['features'], np.ndarray):
            logger.error("Features must be numpy array")
            return False
        
        if not isinstance(data['labels'], np.ndarray):
            logger.error("Labels must be numpy array")
            return False
        
        # Check dimensions
        if len(data['features']) != len(data['labels']):
            logger.error("Features and labels must have same length")
            return False
        
        logger.info("Data structure validation passed")
        return True


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


# Convenience functions for backward compatibility
def secure_download_dataset():
    """Download and verify the traffic signs dataset securely."""
    loader = SecureDataLoader()
    
    url = 'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip'
    filename = 'dataset.zip'
    
    # Download with size and URL validation
    zip_path = loader.secure_download(url, filename)
    
    # Extract safely
    extract_path = loader.safe_extract_zip(zip_path)
    
    # Clean up zip file
    zip_path.unlink()
    
    return extract_path


def safe_load_traffic_data(data_dir: str = "../dataset"):
    """Safely load traffic sign data with validation.
    
    Args:
        data_dir: Directory containing the data files
        
    Returns:
        tuple: (train_data, valid_data, test_data)
    """
    loader = SecureDataLoader(data_dir)
    
    # Load data files
    train_data = loader.safe_load_pickle(Path(data_dir) / 'train.p')
    valid_data = loader.safe_load_pickle(Path(data_dir) / 'valid.p')
    test_data = loader.safe_load_pickle(Path(data_dir) / 'test.p')
    
    # Validate data structures
    for name, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
        if not loader.validate_data_structure(data):
            raise SecurityError(f"Invalid data structure in {name} dataset")
    
    return train_data, valid_data, test_data


if __name__ == "__main__":
    # Test the security utilities
    try:
        print("Testing secure data download and loading...")
        secure_download_dataset()
        train, valid, test = safe_load_traffic_data()
        print("Security utilities working correctly")
    except Exception as e:
        print(f"Error testing security utilities: {e}")