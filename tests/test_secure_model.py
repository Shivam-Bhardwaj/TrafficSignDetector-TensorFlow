#!/usr/bin/env python3
"""
Unit tests for secure_model.py

Tests cover:
- SecureModelHandler functionality
- Model validation
- Secure model loading
- Input validation
- Prediction security
- Error handling
"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import numpy as np

# Skip TensorFlow tests if not available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from secure_model import (
    SecureModelHandler,
    SecurityError,
    secure_model_inference
)


@unittest.skipUnless(TENSORFLOW_AVAILABLE, "TensorFlow not available")
class TestSecureModelHandler(unittest.TestCase):
    """Test cases for SecureModelHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.handler = SecureModelHandler()
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.handler.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_mock_model_files(self):
        """Create mock model files for testing."""
        model_dir = Path(self.temp_dir)
        
        # Create mock checkpoint files
        (model_dir / "model.meta").write_bytes(b"mock meta content")
        (model_dir / "model.index").write_bytes(b"mock index content")
        (model_dir / "model.data-00000-of-00001").write_bytes(b"mock data content")
        
        return model_dir
    
    def test_initialization(self):
        """Test SecureModelHandler initialization."""
        self.assertIsNone(self.handler.session)
        self.assertIsNone(self.handler.model)
        self.assertIsNone(self.handler.model_path)
    
    def test_validate_model_files_valid_checkpoint(self):
        """Test model file validation with valid checkpoint files."""
        model_dir = self.create_mock_model_files()
        
        result = self.handler.validate_model_files(model_dir)
        self.assertTrue(result)
    
    def test_validate_model_files_nonexistent_directory(self):
        """Test model file validation with nonexistent directory."""
        nonexistent_dir = Path(self.temp_dir) / "nonexistent"
        
        result = self.handler.validate_model_files(nonexistent_dir)
        self.assertFalse(result)
    
    def test_validate_model_files_no_valid_files(self):
        """Test model file validation with no valid model files."""
        model_dir = Path(self.temp_dir)
        (model_dir / "random.txt").write_text("not a model file")
        
        result = self.handler.validate_model_files(model_dir)
        self.assertFalse(result)
    
    def test_validate_model_files_oversized_file(self):
        """Test model file validation rejects oversized files."""
        model_dir = Path(self.temp_dir)
        
        # Mock file size check
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_size = self.handler.MAX_MODEL_SIZE + 1
            
            # Create a .meta file that would normally be valid
            (model_dir / "model.meta").touch()
            
            result = self.handler.validate_model_files(model_dir)
            self.assertFalse(result)
    
    def test_calculate_model_hash(self):
        """Test model hash calculation."""
        model_dir = self.create_mock_model_files()
        
        hash_result = self.handler.calculate_model_hash(model_dir)
        
        self.assertIsInstance(hash_result, str)
        self.assertEqual(len(hash_result), 64)  # SHA256 length
    
    def test_validate_model_inputs_valid(self):
        """Test input validation with valid data."""
        # Valid image data
        valid_input = np.random.rand(10, 32, 32, 1).astype(np.float32)
        valid_input = (valid_input - 0.5) * 2  # Scale to [-1, 1]
        
        result = self.handler.validate_model_inputs(valid_input)
        self.assertTrue(result)
    
    def test_validate_model_inputs_wrong_type(self):
        """Test input validation rejects non-numpy arrays."""
        invalid_input = [[1, 2, 3], [4, 5, 6]]
        
        result = self.handler.validate_model_inputs(invalid_input)
        self.assertFalse(result)
    
    def test_validate_model_inputs_extreme_values(self):
        """Test input validation rejects extreme values."""
        # Values outside reasonable range
        invalid_input = np.array([[100.0, -100.0]], dtype=np.float32)
        
        result = self.handler.validate_model_inputs(invalid_input)
        self.assertFalse(result)
    
    def test_validate_model_inputs_nan_values(self):
        """Test input validation rejects NaN values."""
        invalid_input = np.array([[1.0, np.nan, 2.0]], dtype=np.float32)
        
        result = self.handler.validate_model_inputs(invalid_input)
        self.assertFalse(result)
    
    def test_validate_model_inputs_infinite_values(self):
        """Test input validation rejects infinite values."""
        invalid_input = np.array([[1.0, np.inf, 2.0]], dtype=np.float32)
        
        result = self.handler.validate_model_inputs(invalid_input)
        self.assertFalse(result)
    
    def test_validate_model_inputs_shape_mismatch(self):
        """Test input validation with shape mismatch."""
        input_data = np.random.rand(10, 28, 28, 1).astype(np.float32)  # Wrong shape
        expected_shape = (10, 32, 32, 1)
        
        result = self.handler.validate_model_inputs(input_data, expected_shape)
        self.assertFalse(result)
    
    def test_validate_model_inputs_too_large(self):
        """Test input validation rejects oversized data."""
        # Create data that exceeds memory limit
        oversized_input = np.random.rand(10000, 1000, 1000, 3).astype(np.float32)
        
        result = self.handler.validate_model_inputs(oversized_input)
        self.assertFalse(result)
    
    @patch('tensorflow.compat.v1.Session')
    @patch('tensorflow.compat.v1.train.import_meta_graph')
    def test_safe_restore_model_success(self, mock_import_meta, mock_session_class):
        """Test successful model restoration."""
        # Setup mocks
        model_dir = self.create_mock_model_files()
        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_saver = MagicMock()
        mock_import_meta.return_value = mock_saver
        
        # Test restoration
        result_session = self.handler.safe_restore_model(str(model_dir))
        
        # Verify results
        self.assertEqual(result_session, mock_session)
        mock_import_meta.assert_called_once()
        mock_saver.restore.assert_called_once()
    
    def test_safe_restore_model_invalid_directory(self):
        """Test model restoration with invalid directory."""
        invalid_dir = Path(self.temp_dir) / "nonexistent"
        
        with self.assertRaises(SecurityError) as context:
            self.handler.safe_restore_model(str(invalid_dir))
        
        self.assertIn("validation failed", str(context.exception))
    
    def test_safe_restore_model_hash_mismatch(self):
        """Test model restoration with hash mismatch."""
        model_dir = self.create_mock_model_files()
        wrong_hash = "0" * 64  # Obviously wrong hash
        
        with self.assertRaises(SecurityError) as context:
            self.handler.safe_restore_model(str(model_dir), wrong_hash)
        
        self.assertIn("hash mismatch", str(context.exception).lower())
    
    @patch('tensorflow.compat.v1.Session')
    def test_safe_predict_no_model_loaded(self, mock_session_class):
        """Test prediction without loaded model."""
        input_data = np.random.rand(1, 32, 32, 1).astype(np.float32)
        
        with self.assertRaises(SecurityError) as context:
            self.handler.safe_predict(input_data)
        
        self.assertIn("No model loaded", str(context.exception))
    
    @patch('tensorflow.compat.v1.Session')
    def test_safe_predict_invalid_input(self, mock_session_class):
        """Test prediction with invalid input."""
        # Setup mock session
        self.handler.session = MagicMock()
        
        # Invalid input (wrong type)
        invalid_input = [[1, 2, 3]]
        
        with self.assertRaises(SecurityError) as context:
            self.handler.safe_predict(invalid_input)
        
        self.assertIn("Input validation failed", str(context.exception))
    
    def test_cleanup(self):
        """Test resource cleanup."""
        # Setup mock session
        mock_session = MagicMock()
        self.handler.session = mock_session
        
        # Cleanup
        self.handler.cleanup()
        
        # Verify cleanup
        mock_session.close.assert_called_once()
        self.assertIsNone(self.handler.session)


class TestSecureModelUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('secure_model.SecureModelHandler')
    def test_secure_model_inference(self, mock_handler_class):
        """Test secure model inference function."""
        # Setup mocks
        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler
        
        mock_predictions = np.array([[0.1, 0.9], [0.8, 0.2]])
        mock_handler.safe_predict.return_value = mock_predictions
        
        # Test data
        input_data = np.random.rand(2, 32, 32, 1).astype(np.float32)
        
        # Test inference
        result = secure_model_inference(self.temp_dir, input_data)
        
        # Verify results
        self.assertTrue(np.array_equal(result, mock_predictions))
        mock_handler.safe_restore_model.assert_called_once_with(self.temp_dir, None)
        mock_handler.safe_predict.assert_called_once_with(input_data)
        mock_handler.cleanup.assert_called_once()
    
    @patch('secure_model.SecureModelHandler')
    def test_secure_model_inference_with_hash(self, mock_handler_class):
        """Test secure model inference with hash verification."""
        # Setup mocks
        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler
        
        test_hash = "a" * 64
        input_data = np.random.rand(1, 32, 32, 1).astype(np.float32)
        
        # Test inference with hash
        secure_model_inference(self.temp_dir, input_data, test_hash)
        
        # Verify hash was passed
        mock_handler.safe_restore_model.assert_called_once_with(self.temp_dir, test_hash)
    
    @patch('secure_model.SecureModelHandler')
    def test_secure_model_inference_cleanup_on_error(self, mock_handler_class):
        """Test that cleanup is called even when errors occur."""
        # Setup mocks
        mock_handler = MagicMock()
        mock_handler_class.return_value = mock_handler
        mock_handler.safe_restore_model.side_effect = Exception("Test error")
        
        input_data = np.random.rand(1, 32, 32, 1).astype(np.float32)
        
        # Test that error propagates but cleanup is still called
        with self.assertRaises(Exception):
            secure_model_inference(self.temp_dir, input_data)
        
        # Verify cleanup was called despite error
        mock_handler.cleanup.assert_called_once()


class TestSecurityErrorHandling(unittest.TestCase):
    """Test cases for security error handling."""
    
    def test_security_error_creation(self):
        """Test SecurityError exception creation."""
        error_message = "Test security error"
        error = SecurityError(error_message)
        
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), error_message)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.handler = SecureModelHandler()
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.handler.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_empty_model_directory_hash(self):
        """Test hash calculation for empty model directory."""
        empty_dir = Path(self.temp_dir)
        
        hash_result = self.handler.calculate_model_hash(empty_dir)
        self.assertIsInstance(hash_result, str)
        self.assertEqual(len(hash_result), 64)  # SHA256 length
    
    def test_model_files_with_unicode_names(self):
        """Test handling of model files with Unicode names."""
        model_dir = Path(self.temp_dir)
        unicode_filename = "模型.meta"
        (model_dir / unicode_filename).write_bytes(b"mock content")
        
        # Should handle Unicode filenames gracefully
        hash_result = self.handler.calculate_model_hash(model_dir)
        self.assertIsInstance(hash_result, str)
    
    def test_validate_inputs_boundary_values(self):
        """Test input validation at boundary values."""
        # Test exactly at the memory limit
        # Calculate size that's just under the limit
        max_memory = 100 * 1024 * 1024  # 100MB
        elements_per_mb = 1024 * 1024 // 4  # 4 bytes per float32
        max_elements = max_memory // 4
        
        # Create data just under the limit
        safe_size = max_elements - 1000  # Slightly under limit
        boundary_input = np.random.rand(safe_size).astype(np.float32) * 2 - 1  # [-1, 1] range
        
        result = self.handler.validate_model_inputs(boundary_input)
        self.assertTrue(result)
    
    def test_multiple_cleanup_calls(self):
        """Test multiple cleanup calls don't cause errors."""
        # Should handle multiple cleanup calls gracefully
        self.handler.cleanup()
        self.handler.cleanup()  # Should not raise error
        
        # Verify state
        self.assertIsNone(self.handler.session)


class TestSecurityIntegration(unittest.TestCase):
    """Integration tests for security features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_security_validation(self):
        """Test end-to-end security validation flow."""
        handler = SecureModelHandler()
        
        try:
            # Test sequence of security validations
            model_dir = Path(self.temp_dir)
            
            # 1. Validate empty directory (should fail)
            self.assertFalse(handler.validate_model_files(model_dir))
            
            # 2. Create invalid files (should still fail)
            (model_dir / "not_a_model.txt").write_text("invalid")
            self.assertFalse(handler.validate_model_files(model_dir))
            
            # 3. Create valid model files (should pass)
            (model_dir / "model.meta").write_bytes(b"mock meta")
            (model_dir / "model.index").write_bytes(b"mock index")
            self.assertTrue(handler.validate_model_files(model_dir))
            
            # 4. Calculate hash (should work)
            hash_result = handler.calculate_model_hash(model_dir)
            self.assertIsInstance(hash_result, str)
            self.assertEqual(len(hash_result), 64)
            
            # 5. Test input validation
            valid_input = np.random.rand(1, 32, 32, 1).astype(np.float32)
            self.assertTrue(handler.validate_model_inputs(valid_input))
            
            invalid_input = np.array([[np.nan]])
            self.assertFalse(handler.validate_model_inputs(invalid_input))
            
        finally:
            handler.cleanup()


if __name__ == '__main__':
    # Configure test runner
    unittest.main(
        verbosity=2,
        failfast=False,
        buffer=True,
        warnings='ignore'
    )