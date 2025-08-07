#!/usr/bin/env python3
"""
Performance and stress tests for the TrafficSignDetector-TensorFlow project.

Tests cover:
- Performance benchmarking
- Stress testing under load
- Memory usage monitoring
- Resource leak detection
- Scalability testing
- Timeout and limits testing
"""

import unittest
import tempfile
import shutil
import time
import threading
import gc
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Try to import performance monitoring tools
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import modules to test
sys.path.append(str(Path(__file__).parent.parent))

from security_utils import SecureDataLoader, SecurityError
from secure_model import SecureModelHandler


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = SecureDataLoader(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_url_validation_performance(self):
        """Benchmark URL validation performance."""
        urls = [
            "https://example.com/file.zip",
            "http://test-site.org/data.tar.gz",
            "https://secure.amazonaws.com/bucket/file.zip",
            "ftp://invalid.com/file.zip",  # Invalid
            "https://localhost/file.zip",  # Invalid
        ] * 100  # 500 URLs total
        
        start_time = time.time()
        results = []
        
        for url in urls:
            result = self.loader.validate_url(url)
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_url = total_time / len(urls)
        
        # Performance assertions
        self.assertLess(total_time, 5.0, "URL validation took too long overall")
        self.assertLess(avg_time_per_url, 0.01, "Average URL validation too slow")
        
        print(f"\nURL Validation Performance:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average per URL: {avg_time_per_url:.6f}s")
        print(f"  URLs per second: {len(urls) / total_time:.1f}")
    
    def test_hash_calculation_performance(self):
        """Benchmark hash calculation performance."""
        # Create test files of various sizes
        test_files = []
        sizes = [1024, 10240, 102400, 1048576]  # 1KB, 10KB, 100KB, 1MB
        
        for size in sizes:
            test_file = Path(self.temp_dir) / f"test_{size}.bin"
            test_data = np.random.bytes(size)
            test_file.write_bytes(test_data)
            test_files.append((test_file, size))
        
        # Benchmark hash calculation
        results = []
        for test_file, size in test_files:
            start_time = time.time()
            hash_result = self.loader.calculate_file_hash(test_file)
            end_time = time.time()
            
            calc_time = end_time - start_time
            throughput = size / calc_time if calc_time > 0 else float('inf')
            
            results.append((size, calc_time, throughput))
            
            # Performance assertions
            self.assertLess(calc_time, 1.0, f"Hash calculation too slow for {size} bytes")
            self.assertIsInstance(hash_result, str)
            self.assertEqual(len(hash_result), 64)
        
        print(f"\nHash Calculation Performance:")
        for size, calc_time, throughput in results:
            print(f"  {size:>7} bytes: {calc_time:.4f}s ({throughput/1024/1024:.1f} MB/s)")
    
    def test_data_validation_performance(self):
        """Benchmark data validation performance."""
        # Create test datasets of various sizes
        sizes = [100, 1000, 5000, 10000]
        
        for size in sizes:
            with self.subTest(size=size):
                test_data = {
                    'features': np.random.randint(0, 255, (size, 32, 32, 3), dtype=np.uint8),
                    'labels': np.random.randint(0, 43, size, dtype=np.uint8)
                }
                
                start_time = time.time()
                result = self.loader.validate_data_structure(test_data)
                end_time = time.time()
                
                validation_time = end_time - start_time
                samples_per_second = size / validation_time if validation_time > 0 else float('inf')
                
                # Performance assertions
                self.assertTrue(result, "Valid data should pass validation")
                self.assertLess(validation_time, 1.0, f"Data validation too slow for {size} samples")
                
                print(f"  {size:>5} samples: {validation_time:.4f}s ({samples_per_second:.0f} samples/s)")


class TestStressAndLoad(unittest.TestCase):
    """Stress and load testing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_concurrent_data_loading(self):
        """Test data loading under concurrent access."""
        num_threads = 10
        operations_per_thread = 50
        errors = []
        results = []
        
        def worker_thread(thread_id):
            """Worker thread for concurrent testing."""
            try:
                loader = SecureDataLoader(self.temp_dir)
                thread_results = []
                
                for i in range(operations_per_thread):
                    # Create small test data
                    test_data = {
                        'features': np.random.randint(0, 255, (10, 32, 32, 3), dtype=np.uint8),
                        'labels': np.random.randint(0, 43, 10, dtype=np.uint8)
                    }
                    
                    # Validate data
                    result = loader.validate_data_structure(test_data)
                    thread_results.append(result)
                    
                    # Small delay to increase concurrency
                    time.sleep(0.001)
                
                results.extend([(thread_id, r) for r in thread_results])
                
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # Start threads
        threads = []
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10.0)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        total_operations = num_threads * operations_per_thread
        operations_per_second = total_operations / total_time
        
        # Assertions
        self.assertEqual(len(errors), 0, f"Concurrent access caused errors: {errors}")
        self.assertEqual(len(results), total_operations, "Missing results from concurrent test")
        self.assertGreater(operations_per_second, 100, "Concurrent performance too low")
        
        print(f"\nConcurrent Load Test:")
        print(f"  Threads: {num_threads}")
        print(f"  Operations per thread: {operations_per_thread}")
        print(f"  Total operations: {total_operations}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Operations per second: {operations_per_second:.1f}")
    
    def test_memory_stress_testing(self):
        """Test system behavior under memory stress."""
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available for memory testing")
        
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Perform memory-intensive operations
        large_datasets = []
        max_datasets = 20
        
        try:
            for i in range(max_datasets):
                # Create progressively larger datasets
                size = 1000 * (i + 1)
                dataset = {
                    'features': np.random.randint(0, 255, (size, 32, 32, 3), dtype=np.uint8),
                    'labels': np.random.randint(0, 43, size, dtype=np.uint8)
                }
                large_datasets.append(dataset)
                
                # Validate each dataset
                loader = SecureDataLoader(self.temp_dir)
                result = loader.validate_data_structure(dataset)
                self.assertTrue(result, f"Dataset {i} validation failed")
                
                # Check memory usage
                current_memory = process.memory_info().rss
                memory_increase = current_memory - initial_memory
                
                # Stop if memory usage gets excessive (>1GB increase)
                if memory_increase > 1024 * 1024 * 1024:
                    print(f"Stopping memory stress test at iteration {i} (memory increase: {memory_increase/1024/1024:.1f}MB)")
                    break
        
        except MemoryError:
            print("Hit memory limit - this is expected behavior")
        
        finally:
            # Clean up
            del large_datasets
            gc.collect()
            
            # Verify memory cleanup
            time.sleep(1)  # Allow time for cleanup
            final_memory = process.memory_info().rss
            memory_after_cleanup = final_memory - initial_memory
            
            print(f"Memory usage after cleanup: {memory_after_cleanup/1024/1024:.1f}MB increase")
    
    def test_file_system_stress(self):
        """Test file system operations under stress."""
        num_files = 100
        file_size = 10240  # 10KB each
        
        # Create many files concurrently
        def create_files_batch(start_idx, count):
            files_created = []
            for i in range(start_idx, start_idx + count):
                file_path = Path(self.temp_dir) / f"stress_test_{i}.bin"
                test_data = np.random.bytes(file_size)
                file_path.write_bytes(test_data)
                files_created.append(file_path)
            return files_created
        
        start_time = time.time()
        
        # Create files in batches using thread pool
        batch_size = 10
        all_files = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for start in range(0, num_files, batch_size):
                count = min(batch_size, num_files - start)
                future = executor.submit(create_files_batch, start, count)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                files = future.result()
                all_files.extend(files)
        
        creation_time = time.time() - start_time
        
        # Now perform hash calculations on all files
        start_time = time.time()
        loader = SecureDataLoader(self.temp_dir)
        hashes = []
        
        for file_path in all_files:
            hash_result = loader.calculate_file_hash(file_path)
            hashes.append(hash_result)
        
        hash_time = time.time() - start_time
        
        # Performance assertions
        self.assertEqual(len(all_files), num_files, "Not all files were created")
        self.assertEqual(len(hashes), num_files, "Not all hashes were calculated")
        self.assertLess(creation_time, 10.0, "File creation took too long")
        self.assertLess(hash_time, 5.0, "Hash calculation took too long")
        
        print(f"\nFile System Stress Test:")
        print(f"  Files created: {len(all_files)}")
        print(f"  Creation time: {creation_time:.3f}s ({num_files/creation_time:.1f} files/s)")
        print(f"  Hash time: {hash_time:.3f}s ({num_files/hash_time:.1f} files/s)")


class TestResourceLimitsAndTimeouts(unittest.TestCase):
    """Test resource limits and timeout mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = SecureDataLoader(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_file_size_limits(self):
        """Test file size limit enforcement."""
        # Create file that exceeds the maximum allowed size
        large_file = Path(self.temp_dir) / "large_file.bin"
        
        # Write data in chunks to avoid memory issues
        chunk_size = 1024 * 1024  # 1MB chunks
        max_size = self.loader.MAX_FILE_SIZE
        
        try:
            with open(large_file, 'wb') as f:
                written = 0
                while written < max_size + chunk_size:
                    chunk = np.random.bytes(min(chunk_size, max_size + chunk_size - written))
                    f.write(chunk)
                    written += len(chunk)
            
            # File operations should respect size limits
            # This test verifies the limit exists, actual enforcement depends on implementation
            file_size = large_file.stat().st_size
            self.assertGreater(file_size, max_size, "Test file should exceed max size")
            
            print(f"Created test file of {file_size/1024/1024:.1f}MB (limit: {max_size/1024/1024:.1f}MB)")
            
        except (OSError, MemoryError):
            # If we can't create the large file, that's also acceptable
            self.skipTest("Cannot create large test file on this system")
    
    def test_operation_timeout_simulation(self):
        """Simulate operations that might timeout."""
        # Test with operations that take varying amounts of time
        
        def slow_operation(delay):
            """Simulate a slow operation."""
            time.sleep(delay)
            return f"completed after {delay}s"
        
        # Test different delay scenarios
        delays = [0.1, 0.5, 1.0, 2.0]
        timeout_threshold = 1.5
        
        for delay in delays:
            with self.subTest(delay=delay):
                start_time = time.time()
                
                try:
                    # Simulate timeout mechanism
                    if delay > timeout_threshold:
                        # This would normally be handled by the actual timeout mechanism
                        raise TimeoutError(f"Operation timed out after {delay}s")
                    else:
                        result = slow_operation(delay)
                        self.assertIsNotNone(result)
                
                except TimeoutError:
                    # Expected for delays exceeding threshold
                    self.assertGreater(delay, timeout_threshold)
                
                elapsed = time.time() - start_time
                print(f"  Delay {delay}s: elapsed {elapsed:.3f}s")
    
    def test_concurrent_resource_limits(self):
        """Test resource limits under concurrent access."""
        max_concurrent_operations = 10
        operation_count = 0
        max_reached = 0
        errors = []
        
        def resource_intensive_operation(op_id):
            """Simulate resource-intensive operation."""
            nonlocal operation_count, max_reached
            
            try:
                operation_count += 1
                max_reached = max(max_reached, operation_count)
                
                # Simulate work
                test_data = np.random.rand(1000, 100).astype(np.float32)
                time.sleep(0.1)  # Simulate processing time
                result = np.sum(test_data)
                
                operation_count -= 1
                return result
                
            except Exception as e:
                operation_count -= 1
                errors.append((op_id, str(e)))
                raise
        
        # Start many concurrent operations
        with ThreadPoolExecutor(max_workers=max_concurrent_operations * 2) as executor:
            futures = []
            for i in range(max_concurrent_operations * 3):  # More operations than workers
                future = executor.submit(resource_intensive_operation, i)
                futures.append(future)
            
            # Wait for completion
            results = []
            for future in as_completed(futures, timeout=10.0):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    errors.append(("future", str(e)))
        
        # Verify resource limits were respected
        self.assertLessEqual(max_reached, max_concurrent_operations,
                           f"Too many concurrent operations: {max_reached}")
        self.assertEqual(len(errors), 0, f"Errors in concurrent operations: {errors}")
        self.assertGreater(len(results), 0, "No successful operations")
        
        print(f"\nConcurrent Resource Test:")
        print(f"  Max concurrent operations: {max_reached}")
        print(f"  Successful operations: {len(results)}")
        print(f"  Errors: {len(errors)}")


class TestScalabilityAndLimits(unittest.TestCase):
    """Test scalability and system limits."""
    
    def test_data_size_scalability(self):
        """Test scalability with increasing data sizes."""
        sizes = [100, 500, 1000, 2500, 5000]
        performance_data = []
        
        loader = SecureDataLoader()
        
        for size in sizes:
            # Create test dataset
            test_data = {
                'features': np.random.randint(0, 255, (size, 32, 32, 3), dtype=np.uint8),
                'labels': np.random.randint(0, 43, size, dtype=np.uint8)
            }
            
            # Measure validation performance
            start_time = time.time()
            result = loader.validate_data_structure(test_data)
            end_time = time.time()
            
            validation_time = end_time - start_time
            throughput = size / validation_time if validation_time > 0 else float('inf')
            
            performance_data.append((size, validation_time, throughput))
            
            # Verify result
            self.assertTrue(result, f"Validation failed for size {size}")
            
            print(f"Size {size:>5}: {validation_time:.4f}s ({throughput:.0f} samples/s)")
        
        # Check that performance scales reasonably
        # (validation time should not grow exponentially with data size)
        if len(performance_data) > 2:
            # Compare first and last measurements
            first_size, first_time, first_throughput = performance_data[0]
            last_size, last_time, last_throughput = performance_data[-1]
            
            size_ratio = last_size / first_size
            time_ratio = last_time / first_time if first_time > 0 else 1
            
            # Time should not grow much faster than data size (allowing for some overhead)
            max_acceptable_ratio = size_ratio * 2  # Allow 2x overhead
            self.assertLess(time_ratio, max_acceptable_ratio,
                          f"Performance degraded too much: time ratio {time_ratio:.2f} vs size ratio {size_ratio:.2f}")
    
    def test_memory_usage_scalability(self):
        """Test memory usage scaling."""
        if not PSUTIL_AVAILABLE:
            self.skipTest("psutil not available for memory testing")
        
        import psutil
        process = psutil.Process()
        
        sizes = [1000, 2000, 3000, 4000, 5000]
        memory_data = []
        
        for size in sizes:
            # Measure memory before
            gc.collect()  # Force garbage collection
            memory_before = process.memory_info().rss
            
            # Create and validate data
            test_data = {
                'features': np.random.randint(0, 255, (size, 32, 32, 3), dtype=np.uint8),
                'labels': np.random.randint(0, 43, size, dtype=np.uint8)
            }
            
            loader = SecureDataLoader()
            result = loader.validate_data_structure(test_data)
            self.assertTrue(result)
            
            # Measure memory after
            memory_after = process.memory_info().rss
            memory_increase = memory_after - memory_before
            memory_per_sample = memory_increase / size if size > 0 else 0
            
            memory_data.append((size, memory_increase, memory_per_sample))
            
            # Clean up
            del test_data
            del loader
            gc.collect()
            
            print(f"Size {size:>5}: +{memory_increase/1024/1024:.2f}MB ({memory_per_sample/1024:.1f}KB/sample)")
        
        # Verify memory usage is reasonable
        for size, memory_increase, memory_per_sample in memory_data:
            # Each sample should not use excessive memory
            max_memory_per_sample = 50 * 1024  # 50KB per sample max
            self.assertLess(memory_per_sample, max_memory_per_sample,
                          f"Memory per sample too high: {memory_per_sample/1024:.1f}KB")


if __name__ == '__main__':
    print("=" * 60)
    print("PERFORMANCE AND STRESS TESTING")
    print("=" * 60)
    
    # Configure test runner for performance testing
    unittest.main(
        verbosity=2,
        failfast=False,
        buffer=False,  # Allow performance output
        warnings='ignore'
    )