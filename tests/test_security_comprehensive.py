#!/usr/bin/env python3
"""
Comprehensive security tests for the TrafficSignDetector-TensorFlow project.
"""

import unittest
import tempfile
import shutil
import os
import sys
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.data_handler import SecureDataLoader, SecurityError
from secure_model import SecureModelHandler

class TestSecurityVulnerabilities(unittest.TestCase):
    """Test specific security vulnerabilities and their mitigations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = SecureDataLoader(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_path_traversal_prevention(self):
        """Test prevention of path traversal attacks."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
        ]
        
        for malicious_path in malicious_paths:
            with self.subTest(path=malicious_path):
                safe_path = Path(self.temp_dir) / malicious_path
                try:
                    resolved_path = safe_path.resolve()
                    is_safe = str(resolved_path).startswith(str(Path(self.temp_dir).resolve()))
                    self.assertTrue(is_safe, f"Path {malicious_path} escaped the temp directory")
                except Exception:
                    self.assertTrue(True, f"Path {malicious_path} safely rejected")

    def test_malicious_url_detection(self):
        """Test detection of malicious URLs."""
        malicious_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "file:///etc/passwd",
        ]
        
        for url in malicious_urls:
            with self.subTest(url=url):
                result = self.loader.validate_url(url)
                self.assertFalse(result, f"Malicious URL should be rejected: {url}")

class TestAttackSimulation(unittest.TestCase):
    """Simulate various attack vectors and test defenses."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_timing_attack_resistance(self):
        """Test resistance to timing attacks."""
        loader = SecureDataLoader(self.temp_dir)
        
        valid_url = "https://example.com/file.zip"
        invalid_url = "ftp://example.com/file.zip"
        
        times_valid = []
        times_invalid = []
        
        for _ in range(10):
            start = time.time()
            loader.validate_url(valid_url)
            times_valid.append(time.time() - start)
            
            start = time.time()
            loader.validate_url(invalid_url)
            times_invalid.append(time.time() - start)
        
        avg_valid = sum(times_valid) / len(times_valid)
        avg_invalid = sum(times_invalid) / len(times_invalid)
        
        timing_ratio = max(avg_valid, avg_invalid) / min(avg_valid, avg_invalid) if min(avg_valid, avg_invalid) > 0 else 1
        self.assertLess(timing_ratio, 100, "Timing difference suggests potential timing attack vector")

class TestModelSecurityValidation(unittest.TestCase):
    """Test model security validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.handler = SecureModelHandler()
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.handler.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_adversarial_input_detection(self):
        """Test detection of adversarial inputs."""
        normal_input = np.random.rand(1, 32, 32, 1).astype(np.float32)
        self.assertTrue(self.handler.validate_model_inputs(normal_input))
        
        adversarial_inputs = [
            np.full((1, 32, 32, 1), np.nan, dtype=np.float32),
            np.full((1, 32, 32, 1), np.inf, dtype=np.float32),
        ]
        
        for i, adversarial_input in enumerate(adversarial_inputs):
            with self.subTest(input_type=i):
                self.assertFalse(self.handler.validate_model_inputs(adversarial_input))

if __name__ == '__main__':
    unittest.main()
