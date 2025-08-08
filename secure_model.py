#!/usr/bin/env python3
"""
Secure model loading and validation utilities for TensorFlow models.
Provides safe model loading with integrity checks and validation.
"""

import os
import hashlib
from pathlib import Path
from typing import Optional
import logging
import tensorflow as tf
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureModelHandler:
    """Secure TensorFlow model loading and validation."""
    
    MAX_MODEL_SIZE = 1 * 1024 * 1024 * 1024  # 1GB max model size
    
    def __init__(self):
        """Initialize secure model handler."""
        self.session = None
    
    def validate_model_files(self, model_dir: Path) -> bool:
        """Validate model files for security."""
        if not model_dir.exists():
            return False

        try:
            if not any(model_dir.iterdir()):
                return False
        except OSError:
            return False

        required_suffixes = ['.meta', '.index']
        for suffix in required_suffixes:
            if not any(f.suffix == suffix for f in model_dir.iterdir()):
                return False
        
        if not any('.data-00000-of-00001' in f.name for f in model_dir.iterdir()):
            return False

        for file_path in model_dir.iterdir():
            if file_path.is_file() and file_path.stat().st_size > self.MAX_MODEL_SIZE:
                logger.error(f"Model file too large: {file_path}")
                return False
        
        return True

    def calculate_model_hash(self, model_dir: Path) -> str:
        """Calculate hash of all model files for integrity verification."""
        hash_obj = hashlib.sha256()
        model_files = sorted([f for f in model_dir.iterdir() if f.is_file()])
        
        for file_path in model_files:
            hash_obj.update(file_path.name.encode('utf-8'))
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
        return hash_obj.hexdigest()
    
    def safe_restore_model(self, model_dir: str, expected_hash: Optional[str] = None) -> tf.compat.v1.Session:
        """Safely restore TensorFlow model with validation."""
        model_dir_path = Path(model_dir).resolve()
        
        if not self.validate_model_files(model_dir_path):
            raise SecurityError(f"Model validation failed for: {model_dir}")
        
        if expected_hash:
            actual_hash = self.calculate_model_hash(model_dir_path)
            if actual_hash != expected_hash:
                raise SecurityError(f"Model hash mismatch. Expected: {expected_hash}, Got: {actual_hash}")
        
        self.session = tf.compat.v1.Session()
        try:
            meta_file = str(next(model_dir_path.glob("*.meta")))
            saver = tf.compat.v1.train.import_meta_graph(meta_file)
            saver.restore(self.session, tf.train.latest_checkpoint(model_dir))
            return self.session
        except Exception as e:
            if self.session:
                self.session.close()
                self.session = None
            raise SecurityError(f"Model loading failed: {e}")
    
    def validate_model_inputs(self, input_data: np.ndarray) -> bool:
        """Validate model input data for security."""
        if not isinstance(input_data, np.ndarray):
            return False
        if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
            return False
        
        max_memory = 100 * 1024 * 1024 
        if input_data.nbytes > max_memory:
            return False
            
        return True
    
    def cleanup(self):
        """Clean up resources."""
        if self.session:
            self.session.close()
            self.session = None

class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass