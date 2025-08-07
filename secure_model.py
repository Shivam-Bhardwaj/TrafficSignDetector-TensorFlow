#!/usr/bin/env python3
"""
Secure model loading and validation utilities for TensorFlow models.
Provides safe model loading with integrity checks and validation.
"""

import os
import hashlib
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union
import logging

try:
    import tensorflow as tf
    import numpy as np
except ImportError as e:
    print(f"Error: Required dependencies not installed: {e}")
    print("Run: pip install -r requirements.txt")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SecureModelHandler:
    """Secure TensorFlow model loading and validation."""
    
    MAX_MODEL_SIZE = 1 * 1024 * 1024 * 1024  # 1GB max model size
    ALLOWED_MODEL_EXTENSIONS = {'.pb', '.h5', '.hdf5', '.savedmodel', '.ckpt', '.meta', '.index', '.data-00000-of-00001'}
    
    def __init__(self):
        """Initialize secure model handler."""
        self.session = None
        self.model = None
        self.model_path = None
    
    def validate_model_files(self, model_dir: Union[str, Path]) -> bool:
        """Validate model files for security.
        
        Args:
            model_dir: Directory containing model files
            
        Returns:
            bool: True if model files are valid
        """
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            logger.error(f"Model directory does not exist: {model_dir}")
            return False
        
        # Check for required TensorFlow model files
        required_files = []
        has_checkpoint = False
        has_meta = False
        
        for file_path in model_dir.iterdir():
            if not file_path.is_file():
                continue
                
            # Check file size
            if file_path.stat().st_size > self.MAX_MODEL_SIZE:
                logger.error(f"Model file too large: {file_path}")
                return False
            
            # Check file extension
            if file_path.suffix not in self.ALLOWED_MODEL_EXTENSIONS:
                if not any(ext in file_path.name for ext in ['.ckpt', '.meta', '.index', '.data']):
                    logger.warning(f"Unexpected file in model directory: {file_path}")
            
            # Check for TensorFlow checkpoint files
            if '.ckpt' in file_path.name:
                has_checkpoint = True
            if file_path.suffix == '.meta':
                has_meta = True
        
        if has_checkpoint and has_meta:
            logger.info("Valid TensorFlow checkpoint files found")
            return True
        elif any(f.suffix == '.h5' for f in model_dir.iterdir()):
            logger.info("Valid HDF5 model file found")
            return True
        elif (model_dir / 'saved_model.pb').exists():
            logger.info("Valid SavedModel format found")
            return True
        else:
            logger.error("No valid model files found")
            return False
    
    def calculate_model_hash(self, model_dir: Union[str, Path]) -> str:
        """Calculate hash of all model files for integrity verification.
        
        Args:
            model_dir: Directory containing model files
            
        Returns:
            str: SHA256 hash of all model files combined
        """
        model_dir = Path(model_dir)
        hash_obj = hashlib.sha256()
        
        # Sort files for consistent hashing
        model_files = sorted([f for f in model_dir.iterdir() if f.is_file()])
        
        for file_path in model_files:
            # Include filename in hash for additional integrity
            hash_obj.update(file_path.name.encode('utf-8'))
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def safe_restore_model(self, model_dir: str = ".", 
                          expected_hash: Optional[str] = None) -> tf.compat.v1.Session:
        """Safely restore TensorFlow model with validation.
        
        Args:
            model_dir: Directory containing model files
            expected_hash: Expected hash for integrity verification
            
        Returns:
            tf.Session: Restored TensorFlow session
            
        Raises:
            SecurityError: If model fails security checks
        """
        model_dir = Path(model_dir).resolve()
        
        # Validate model files
        if not self.validate_model_files(model_dir):
            raise SecurityError(f"Model validation failed for: {model_dir}")
        
        # Verify hash if provided
        if expected_hash:
            actual_hash = self.calculate_model_hash(model_dir)
            if actual_hash != expected_hash:
                raise SecurityError(f"Model hash mismatch. Expected: {expected_hash}, Got: {actual_hash}")
        
        # Find checkpoint files
        checkpoint_files = list(model_dir.glob("*.meta"))
        if not checkpoint_files:
            raise SecurityError("No .meta file found in model directory")
        
        # Use the most recent checkpoint
        meta_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
        checkpoint_prefix = str(meta_file).replace('.meta', '')
        
        logger.info(f"Loading model from: {checkpoint_prefix}")
        
        # Create secure TensorFlow session
        config = tf.compat.v1.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True  # Prevent GPU memory issues
        
        # Disable dangerous operations
        config.allow_soft_placement = True
        config.experimental.disable_output_partition_graphs = True
        
        # Create session
        self.session = tf.compat.v1.Session(config=config)
        
        try:
            # Import graph
            saver = tf.compat.v1.train.import_meta_graph(str(meta_file))
            
            # Restore weights
            saver.restore(self.session, checkpoint_prefix)
            
            self.model_path = str(model_dir)
            logger.info("Model loaded successfully")
            
            return self.session
            
        except Exception as e:
            # Clean up on failure
            if self.session:
                self.session.close()
                self.session = None
            logger.error(f"Failed to load model: {e}")
            raise SecurityError(f"Model loading failed: {e}")
    
    def validate_model_inputs(self, input_data: np.ndarray, 
                            expected_shape: Optional[tuple] = None) -> bool:
        """Validate model input data for security.
        
        Args:
            input_data: Input data array
            expected_shape: Expected shape of input data
            
        Returns:
            bool: True if input data is valid
        """
        if not isinstance(input_data, np.ndarray):
            logger.error("Input data must be numpy array")
            return False
        
        # Check for reasonable data ranges (for image data)
        if input_data.dtype in [np.float32, np.float64]:
            if np.any(input_data < -10) or np.any(input_data > 10):
                logger.warning("Input data has unusual range, possible adversarial input")
                return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
            logger.error("Input data contains NaN or infinite values")
            return False
        
        # Check expected shape if provided
        if expected_shape and input_data.shape != expected_shape:
            logger.error(f"Input shape mismatch. Expected: {expected_shape}, Got: {input_data.shape}")
            return False
        
        # Check memory usage (prevent DoS)
        memory_size = input_data.nbytes
        max_memory = 100 * 1024 * 1024  # 100MB max
        if memory_size > max_memory:
            logger.error(f"Input data too large: {memory_size} bytes")
            return False
        
        return True
    
    def safe_predict(self, input_data: np.ndarray, 
                    input_tensor_name: str = "x:0",
                    output_tensor_name: str = "accuracy_operation:0") -> np.ndarray:
        """Safely run model prediction with validation.
        
        Args:
            input_data: Input data for prediction
            input_tensor_name: Name of input tensor
            output_tensor_name: Name of output tensor
            
        Returns:
            np.ndarray: Model predictions
        """
        if not self.session:
            raise SecurityError("No model loaded. Call safe_restore_model() first")
        
        # Validate input data
        if not self.validate_model_inputs(input_data):
            raise SecurityError("Input validation failed")
        
        try:
            # Get tensors
            graph = self.session.graph
            input_tensor = graph.get_tensor_by_name(input_tensor_name)
            output_tensor = graph.get_tensor_by_name(output_tensor_name)
            
            # Run prediction with timeout
            feed_dict = {input_tensor: input_data}
            
            # Set operation timeout to prevent DoS
            run_options = tf.compat.v1.RunOptions()
            run_options.timeout_in_ms = 30000  # 30 second timeout
            
            predictions = self.session.run(output_tensor, 
                                         feed_dict=feed_dict,
                                         options=run_options)
            
            # Validate output
            if not isinstance(predictions, np.ndarray):
                raise SecurityError("Model output is not a numpy array")
            
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                raise SecurityError("Model output contains invalid values")
            
            return predictions
            
        except tf.errors.DeadlineExceededError:
            logger.error("Model prediction timed out")
            raise SecurityError("Model prediction timeout")
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise SecurityError(f"Prediction failed: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if self.session:
            self.session.close()
            self.session = None
        logger.info("Model resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()


class SecurityError(Exception):
    """Custom exception for security-related errors."""
    pass


def secure_model_inference(model_dir: str, input_data: np.ndarray,
                         expected_hash: Optional[str] = None) -> np.ndarray:
    """Convenience function for secure model inference.
    
    Args:
        model_dir: Directory containing model files
        input_data: Input data for prediction
        expected_hash: Expected model hash for verification
        
    Returns:
        np.ndarray: Model predictions
    """
    handler = SecureModelHandler()
    
    try:
        # Load model securely
        handler.safe_restore_model(model_dir, expected_hash)
        
        # Run prediction
        predictions = handler.safe_predict(input_data)
        
        return predictions
        
    finally:
        # Always cleanup
        handler.cleanup()


if __name__ == "__main__":
    # Test the secure model handler
    print("Testing secure model handler...")
    
    try:
        handler = SecureModelHandler()
        
        # Test model validation
        if handler.validate_model_files("."):
            print("Model files validation passed")
            
            # Calculate model hash for future verification
            model_hash = handler.calculate_model_hash(".")
            print(f"Model hash: {model_hash}")
            
        else:
            print("No valid model files found")
            
    except Exception as e:
        print(f"Error testing model handler: {e}")
    finally:
        handler.cleanup()