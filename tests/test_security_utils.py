#!/usr/bin/env python3
"""
Unit tests for src/data_handler.py
"""

import unittest
import tempfile
import shutil
import os
import zipfile
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data_handler import (
    SecureDataLoader, 
    SecurityError,
    download_and_extract_dataset,
    load_traffic_data
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
        self.assertEqual(self.loader.base_dir, Path(self.temp_dir))
        self.assertTrue(self.loader.base_dir.exists())
    
    def test_validate_url_valid(self):
        """Test URL validation with valid URLs."""
        self.assertTrue(self.loader.validate_url("https://example.com/file.zip"))
    
    def test_validate_url_invalid(self):
        """Test URL validation with invalid URLs."""
        self.assertFalse(self.loader.validate_url("ftp://example.com/file.zip"))

    def test_calculate_file_hash(self):
        """Test file hash calculation."""
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = b"Hello, World!"
        test_file.write_bytes(test_content)
        
        calculated_hash = self.loader.calculate_file_hash(test_file)
        expected_hash = hashlib.sha256(test_content).hexdigest()
        self.assertEqual(calculated_hash, expected_hash)

    @patch('urllib.request.urlopen')
    def test_secure_download_success(self, mock_urlopen):
        """Test successful secure download."""
        test_content = b"Test file content"
        mock_response = MagicMock()
        mock_response.read.side_effect = [test_content, b'']
        mock_response.headers.get.return_value = str(len(test_content))
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        url = "https://example.com/test.zip"
        filename = "test.zip"
        
        result_path = self.loader.secure_download(url, filename)
        
        self.assertEqual(result_path, Path(self.temp_dir) / filename)
        self.assertTrue(result_path.exists())
        self.assertEqual(result_path.read_bytes(), test_content)

    def test_safe_extract_zip(self):
        """Test safe ZIP extraction."""
        zip_path = Path(self.temp_dir) / "test.zip"
        extract_dir = Path(self.temp_dir) / "extracted"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("file1.txt", "Content 1")
            zf.writestr("subdir/file2.txt", "Content 2")
        
        result_dir = self.loader.safe_extract_zip(zip_path, extract_dir)
        
        self.assertEqual(result_dir, extract_dir)
        self.assertTrue((extract_dir / "file1.txt").exists())
        self.assertTrue((extract_dir / "subdir" / "file2.txt").exists())

    def test_safe_extract_zip_path_traversal(self):
        """Test ZIP extraction blocks path traversal."""
        zip_path = Path(self.temp_dir) / "malicious.zip"
        
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("../../../evil.txt", "Malicious content")
        
        with self.assertRaises(SecurityError):
            self.loader.safe_extract_zip(zip_path)

class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('src.data_handler.SecureDataLoader')
    def test_download_and_extract_dataset(self, mock_loader_class):
        """Test secure dataset download function."""
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        zip_path = Path(self.temp_dir) / "dataset.zip"
        zip_path.touch() # Create the file
        mock_loader.secure_download.return_value = zip_path
        
        download_and_extract_dataset(self.temp_dir)
        
        mock_loader.secure_download.assert_called_once()
        mock_loader.safe_extract_zip.assert_called_once_with(zip_path)

    @patch('src.data_handler.SecureDataLoader')
    def test_load_traffic_data(self, mock_loader_class):
        """Test safe traffic data loading function."""
        mock_loader = MagicMock()
        mock_loader_class.return_value = mock_loader
        
        mock_data = {'features': np.array([]), 'labels': np.array([])}
        mock_loader.safe_load_pickle.return_value = mock_data
        
        train, valid, test = load_traffic_data(self.temp_dir)
        
        self.assertEqual(mock_loader.safe_load_pickle.call_count, 3)
        self.assertEqual(train, mock_data)
        self.assertEqual(valid, mock_data)
        self.assertEqual(test, mock_data)

if __name__ == '__main__':
    unittest.main()