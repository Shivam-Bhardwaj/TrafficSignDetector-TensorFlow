import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import tensorflow as tf
import sys

sys.path.append(str(Path(__file__).parent.parent))

from secure_model import SecureModelHandler, SecurityError

class TestSecureModelHandler(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.handler = SecureModelHandler()

    def tearDown(self):
        self.handler.cleanup()
        shutil.rmtree(self.temp_dir)

    def create_mock_model_files(self):
        model_dir = Path(self.temp_dir)
        (model_dir / "model.meta").write_bytes(b"meta")
        (model_dir / "model.index").write_bytes(b"index")
        (model_dir / "model.data-00000-of-00001").write_bytes(b"data")
        (model_dir / "checkpoint").write_text('model_checkpoint_path: "model"')
        return model_dir

    def test_validate_model_files_valid(self):
        model_dir = self.create_mock_model_files()
        self.assertTrue(self.handler.validate_model_files(model_dir))

    def test_validate_model_files_oversized(self):
        model_dir = self.create_mock_model_files()
        with patch.object(Path, 'stat') as mock_stat:
            mock_stat.return_value.st_size = self.handler.MAX_MODEL_SIZE + 1
            with patch.object(Path, 'is_file', return_value=True):
                self.assertFalse(self.handler.validate_model_files(model_dir))

    def test_validate_model_inputs_too_large(self):
        # Create an array that is larger than the 100MB limit, but not so large that it causes a MemoryError.
        oversized_input = np.random.rand(10, 100, 100, 300).astype(np.float32)
        self.assertFalse(self.handler.validate_model_inputs(oversized_input))

    @patch('tensorflow.compat.v1.train.import_meta_graph')
    @patch('tensorflow.compat.v1.train.Saver')
    def test_safe_restore_model_success(self, mock_saver, mock_import_meta):
        model_dir = self.create_mock_model_files()
        with patch('tensorflow.compat.v1.Session'):
            self.handler.safe_restore_model(str(model_dir))

    def test_safe_restore_model_hash_mismatch(self):
        model_dir = self.create_mock_model_files()
        with self.assertRaises(SecurityError):
            self.handler.safe_restore_model(str(model_dir), expected_hash="wrong_hash")

if __name__ == '__main__':
    unittest.main()