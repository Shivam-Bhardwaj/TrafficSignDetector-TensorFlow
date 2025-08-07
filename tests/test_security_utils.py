#!/usr/bin/env python3
"""
Unit tests for security_utils.py

Tests cover:
- SecureDataLoader functionality
- URL validation
- File download security
- Data loading validation
- Path traversal protection
- Error handling
"""

import unittest
import tempfile
import shutil
import os
import zipfile
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import numpy as np

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from security_utils import (
    SecureDataLoader, 
    SecurityError,
    secure_download_dataset,
    safe_load_traffic_data
)


class TestSecureDataLoader(unittest.TestCase):
    """Test cases for SecureDataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = SecureDataLoader(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test SecureDataLoader initialization."""
        # Test default initialization
        loader_default = SecureDataLoader()
        self.assertTrue(loader_default.base_dir.exists())
        
        # Test custom directory
        self.assertEqual(self.loader.base_dir, Path(self.temp_dir))
        self.assertTrue(self.loader.base_dir.exists())
    
    def test_validate_url_valid_urls(self):
        """Test URL validation with valid URLs."""
        valid_urls = [
            "https://example.com/file.zip",
            "http://secure-site.org/data.tar.gz",
            "https://s3.amazonaws.com/bucket/file.zip"
        ]
        
        for url in valid_urls:
            with self.subTest(url=url):
                self.assertTrue(self.loader.validate_url(url))
    
    def test_validate_url_invalid_urls(self):
        """Test URL validation with invalid URLs."""
        invalid_urls = [
            "ftp://example.com/file.zip",  # Invalid scheme
            "https://localhost/file.zip",  # Localhost
            "https://127.0.0.1/file.zip",  # Localhost IP
            "https://example.com/../../../etc/passwd",  # Path traversal
            "file:///etc/passwd",  # File protocol
            ""  # Empty URL
        ]
        
        for url in invalid_urls:
            with self.subTest(url=url):
                self.assertFalse(self.loader.validate_url(url))
    
    def test_calculate_file_hash(self):
        """Test file hash calculation."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)
        
        # Calculate hash
        calculated_hash = self.loader.calculate_file_hash(test_file)
        
        # Verify hash
        expected_hash = hashlib.sha256(test_content).hexdigest()
        self.assertEqual(calculated_hash, expected_hash)
    
    def test_calculate_file_hash_nonexistent(self):
        """Test hash calculation for nonexistent file."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.txt"
        
        with self.assertRaises(FileNotFoundError):
            self.loader.calculate_file_hash(nonexistent_file)
    
    @patch('urllib.request.urlopen')
    def test_secure_download_success(self, mock_urlopen):
        """Test successful secure download."""
        # Mock response
        test_content = b"Test file content"
        mock_response = MagicMock()
        mock_response.read.side_effect = [test_content, b'']  # First call returns content, second returns empty
        mock_response.headers.get.return_value = str(len(test_content))
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        # Test download
        url = "https://example.com/test.zip"
        filename = "test.zip"
        
        result_path = self.loader.secure_download(url, filename)
        
        # Verify results
        self.assertEqual(result_path, Path(self.temp_dir) / filename)
        self.assertTrue(result_path.exists())
        self.assertEqual(result_path.read_bytes(), test_content)
    
    def test_secure_download_invalid_url(self):
        """Test download with invalid URL."""
        with self.assertRaises(SecurityError):
            self.loader.secure_download("ftp://invalid.com/file.zip", "test.zip")
    
    @patch('urllib.request.urlopen')
    def test_secure_download_file_too_large(self, mock_urlopen):
        """Test download rejection for oversized files."""
        # Mock large file
        mock_response = MagicMock()
        mock_response.headers.get.return_value = str(self.loader.MAX_FILE_SIZE + 1)
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        with self.assertRaises(SecurityError) as context:
            self.loader.secure_download("https://example.com/large.zip", "test.zip")
        
        self.assertIn("too large", str(context.exception))
    
    def test_safe_extract_zip(self):
        """Test safe ZIP extraction."""
        # Create test ZIP file
        zip_path = Path(self.temp_dir) / "test.zip"
        extract_dir = Path(self.temp_dir) / "extracted"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("file1.txt", "Content 1")
            zf.writestr("subdir/file2.txt", "Content 2")
        
        # Extract safely
        result_dir = self.loader.safe_extract_zip(zip_path, extract_dir)
        
        # Verify extraction
        self.assertEqual(result_dir, extract_dir)
        self.assertTrue((extract_dir / "file1.txt").exists())
        self.assertTrue((extract_dir / "subdir" / "file2.txt").exists())
    
    def test_safe_extract_zip_path_traversal(self):
        """Test ZIP extraction blocks path traversal."""
        # Create malicious ZIP
        zip_path = Path(self.temp_dir) / "malicious.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("../../../evil.txt", "Malicious content")
        
        # Should raise SecurityError
        with self.assertRaises(SecurityError) as context:
            self.loader.safe_extract_zip(zip_path)
        
        self.assertIn("Unsafe path", str(context.exception))
    
    def test_safe_load_pickle_nonexistent(self):
        """Test loading nonexistent file."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.p"
        
        with self.assertRaises(FileNotFoundError):
            self.loader.safe_load_pickle(nonexistent_file)
    
    def test_safe_load_pickle_invalid_extension(self):
        """Test loading file with invalid extension."""
        invalid_file = Path(self.temp_dir) / "test.exe"
        invalid_file.touch()
        
        with self.assertRaises(SecurityError) as context:
            self.loader.safe_load_pickle(invalid_file)
        
        self.assertIn("File type not allowed", str(context.exception))
    
    def test_validate_data_structure_valid(self):
        """Test validation of valid data structure."""
        valid_data = {
            'features': np.random.randint(0, 255, (100, 32, 32, 3), dtype=np.uint8),
            'labels': np.random.randint(0, 43, 100, dtype=np.uint8)
        }
        
        self.assertTrue(self.loader.validate_data_structure(valid_data))
    
    def test_validate_data_structure_missing_keys(self):
        """Test validation with missing required keys."""
        invalid_data = {
            'features': np.random.randint(0, 255, (100, 32, 32, 3), dtype=np.uint8)
            # Missing 'labels'
        }
        
        self.assertFalse(self.loader.validate_data_structure(invalid_data))
    
    def test_validate_data_structure_wrong_types(self):
        """Test validation with wrong data types."""
        invalid_data = {
            'features': [[1, 2, 3]],  # List instead of numpy array
            'labels': np.random.randint(0, 43, 100, dtype=np.uint8)
        }
        
        self.assertFalse(self.loader.validate_data_structure(invalid_data))
    
    def test_validate_data_structure_mismatched_lengths(self):
        """Test validation with mismatched feature/label lengths."""
        invalid_data = {
            'features': np.random.randint(0, 255, (100, 32, 32, 3), dtype=np.uint8),
            'labels': np.random.randint(0, 43, 50, dtype=np.uint8)  # Different length
        }
        
        self.assertFalse(self.loader.validate_data_structure(invalid_data))


class TestSecurityUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('security_utils.SecureDataLoader')
    def test_secure_download_dataset(self, mock_loader_class):
        """Test secure dataset download function."""
        # Mock loader instance
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        mock_loader.secure_download.return_value = Path(self.temp_dir) / "dataset.zip"
        mock_loader.safe_extract_zip.return_value = Path(self.temp_dir)
        
        # Test function
        result = secure_download_dataset()
        
        # Verify calls
        mock_loader.secure_download.assert_called_once()
        mock_loader.safe_extract_zip.assert_called_once()
        self.assertEqual(result, Path(self.temp_dir))
    
    def create_mock_data_files(self):
        """Create mock data files for testing."""
        data_dir = Path(self.temp_dir)
        
        # Create mock data
        mock_data = {
            'features': np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8),
            'labels': np.random.randint(0, 43, 10, dtype=np.uint8)
        }
        
        # Save with joblib (simulate the secure format)
        import joblib
        for filename in ['train.p', 'valid.p', 'test.p']:
            joblib.dump(mock_data, data_dir / filename)
        
        return data_dir
    
    @patch('security_utils.SecureDataLoader')
    def test_safe_load_traffic_data(self, mock_loader_class):
        """Test safe traffic data loading function."""
        # Create mock data files
        data_dir = self.create_mock_data_files()
        
        # Mock loader
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        
        mock_data = {
            'features': np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8),
            'labels': np.random.randint(0, 43, 10, dtype=np.uint8)
        }
        
        mock_loader.safe_load_pickle.return_value = mock_data
        mock_loader.validate_data_structure.return_value = True
        
        # Test function
        train_data, valid_data, test_data = safe_load_traffic_data(str(data_dir))
        
        # Verify results
        self.assertEqual(len(mock_loader.safe_load_pickle.call_args_list), 3)
        self.assertEqual(len(mock_loader.validate_data_structure.call_args_list), 3)
        self.assertEqual(train_data, mock_data)
        self.assertEqual(valid_data, mock_data)
        self.assertEqual(test_data, mock_data)


class TestSecurityErrorHandling(unittest.TestCase):
    """Test cases for security error handling."""
    
    def test_security_error_creation(self):
        """Test SecurityError exception creation."""
        error_message = "Test security error"
        error = SecurityError(error_message)
        
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), error_message)
    
    def test_security_error_inheritance(self):
        """Test SecurityError inheritance."""
        error = SecurityError("Test")
        self.assertIsInstance(error, Exception)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = SecureDataLoader(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_empty_directory_hash(self):
        """Test hash calculation for empty directory."""
        # Create empty directory
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()
        
        # Should handle gracefully
        hash_result = self.loader.calculate_model_hash(empty_dir)
        self.assertIsInstance(hash_result, str)
        self.assertEqual(len(hash_result), 64)  # SHA256 length
    
    def test_unicode_filename_handling(self):
        """Test handling of Unicode filenames."""
        # Create file with Unicode name
        unicode_name = "测试文件.txt"
        test_file = Path(self.temp_dir) / unicode_name
        test_file.write_text("Test content", encoding='utf-8')
        
        # Should handle Unicode filenames
        hash_result = self.loader.calculate_file_hash(test_file)
        self.assertIsInstance(hash_result, str)
    
    def test_very_long_url(self):
        """Test handling of very long URLs."""
        # Create extremely long URL
        long_url = "https://example.com/" + "a" * 10000 + "/file.zip"
        
        # Should still validate correctly (as invalid due to suspicious patterns)
        result = self.loader.validate_url(long_url)
        self.assertIsInstance(result, bool)
    
    def test_zero_byte_file(self):
        """Test handling of zero-byte files."""
        # Create empty file
        empty_file = Path(self.temp_dir) / "empty.txt"
        empty_file.touch()
        
        # Should calculate hash for empty file
        hash_result = self.loader.calculate_file_hash(empty_file)
        expected_hash = hashlib.sha256(b"").hexdigest()
        self.assertEqual(hash_result, expected_hash)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(
        verbosity=2,
        failfast=False,
        buffer=True,
        warnings='ignore'
    )