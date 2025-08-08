#!/usr/bin/env python3
"""
Comprehensive test runner for TrafficSignDetector-TensorFlow security testing framework.

This script runs all test suites and provides detailed reporting:
- Security tests
- Unit tests
- Integration tests
- Performance tests
- Stress tests

Usage:
    python test_runner.py [options]

Options:
    --fast          Run only fast tests (skip performance/stress tests)
    --security      Run only security tests
    --performance   Run only performance tests
    --coverage      Generate coverage report
    --verbose       Verbose output
    --parallel      Run tests in parallel (where supported)
    --report        Generate detailed HTML report
"""

import sys
import os
import unittest
import time
import argparse
from pathlib import Path
from io import StringIO
import json
import traceback
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import test modules
try:
    from tests.test_security_utils import *
    from tests.test_secure_model import *
    from tests.test_security_comprehensive import *
    from tests.test_performance_stress import *
    print("All test modules imported successfully")
except ImportError as e:
    print(f"Error importing test modules: {e}")
    print("Make sure all test files are in the 'tests' directory")
    sys.exit(1)

# Try to import optional dependencies
try:
    import coverage
    COVERAGE_AVAILABLE = True
except ImportError:
    COVERAGE_AVAILABLE = False

try:
    import concurrent.futures
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False


class ColoredTextTestResult(unittest.TextTestResult):
    """Custom test result class with colored output."""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.success_count = 0
        self.start_time = None
        self.test_results = []
        self.verbosity = verbosity
    
    def startTest(self, test):
        super().startTest(test)
        self.start_time = time.time()
    
    def addSuccess(self, test):
        super().addSuccess(test)
        self.success_count += 1
        duration = time.time() - self.start_time if self.start_time else 0
        self.test_results.append({
            'test': str(test),
            'status': 'PASS',
            'duration': duration,
            'message': None
        })
        if self.verbosity > 1:
            self.stream.write(f"PASS {test} ({duration:.3f}s)\n")
    
    def addError(self, test, err):
        super().addError(test, err)
        duration = time.time() - self.start_time if self.start_time else 0
        self.test_results.append({
            'test': str(test),
            'status': 'ERROR',
            'duration': duration,
            'message': self._exc_info_to_string(err, test)
        })
        if self.verbosity > 0:
            self.stream.write(f"ERROR {test} ({duration:.3f}s)\n")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        duration = time.time() - self.start_time if self.start_time else 0
        self.test_results.append({
            'test': str(test),
            'status': 'FAIL',
            'duration': duration,
            'message': self._exc_info_to_string(err, test)
        })
        if self.verbosity > 0:
            self.stream.write(f"FAIL {test} ({duration:.3f}s)\n")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        duration = time.time() - self.start_time if self.start_time else 0
        self.test_results.append({
            'test': str(test),
            'status': 'SKIP',
            'duration': duration,
            'message': reason
        })
        if self.verbosity > 0:
            self.stream.write(f"SKIP {test} - {reason} ({duration:.3f}s)\n")


class TestSuiteRunner:
    """Main test suite runner with advanced features."""
    
    def __init__(self, args):
        self.args = args
        self.start_time = None
        self.test_results = {}
        self.coverage_data = None
        
    def setup_coverage(self):
        """Setup code coverage monitoring."""
        if not COVERAGE_AVAILABLE or not self.args.coverage:
            return None
        
        cov = coverage.Coverage(
            source=[str(project_root)],
            omit=[
                '*/tests/*',
                '*/test_*',
                'test_runner.py',
                '*/venv/*',
                '*/virtualenv/*'
            ]
        )
        cov.start()
        return cov
    
    def get_test_suites(self):
        """Get test suites based on command line arguments."""
        suites = {}
        
        if self.args.security or not any([self.args.performance, self.args.fast]):
            # Security and utility tests
            suites['Security Utils'] = unittest.TestLoader().loadTestsFromName('tests.test_security_utils')
            suites['Model Security'] = unittest.TestLoader().loadTestsFromName('tests.test_secure_model')
            suites['Security Comprehensive'] = unittest.TestLoader().loadTestsFromName('tests.test_security_comprehensive')
        
        if self.args.performance or not any([self.args.security, self.args.fast]):
            # Performance tests
            suites['Performance & Stress'] = unittest.TestLoader().loadTestsFromName('tests.test_performance_stress')
        
        return suites
    
    def run_test_suite(self, name, suite):
        """Run a single test suite."""
        print(f"\n{'=' * 60}")
        print(f"RUNNING: {name}")
        print(f"{'=' * 60}")
        
        # Create custom test runner
        stream = StringIO() if not self.args.verbose else sys.stdout
        runner = unittest.TextTestRunner(
            stream=stream,
            verbosity=2 if self.args.verbose else 1,
            resultclass=ColoredTextTestResult,
            buffer=True
        )
        
        # Run tests
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # Collect results
        duration = end_time - start_time
        self.test_results[name] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'success': result.success_count if hasattr(result, 'success_count') else 
                      result.testsRun - len(result.failures) - len(result.errors),
            'duration': duration,
            'test_results': getattr(result, 'test_results', [])
        }
        
        # Print summary for this suite
        print(f"\n{name} Results:")
        print(f"   Tests run: {result.testsRun}")
        print(f"   Successes: {self.test_results[name]['success']}")
        print(f"   Failures: {len(result.failures)}")
        print(f"   Errors: {len(result.errors)}")
        print(f"   Skipped: {len(result.skipped)}")
        print(f"   Duration: {duration:.2f}s")
        
        # Print failures and errors if any
        if result.failures:
            print(f"\nFAILURES in {name}:")
            for test, traceback in result.failures:
                print(f"   • {test}")
                if self.args.verbose:
                    print(f"     {traceback}")
        
        if result.errors:
            print(f"\nERRORS in {name}:")
            for test, traceback in result.errors:
                print(f"   • {test}")
                if self.args.verbose:
                    print(f"     {traceback}")
        
        return result
    
    def run_all_tests(self):
        """Run all selected test suites."""
        print("Starting TrafficSignDetector-TensorFlow Security Test Suite")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Python: {sys.version}")
        print(f"Project root: {project_root}")
        
        # Setup coverage
        cov = self.setup_coverage()
        
        self.start_time = time.time()
        all_results = []
        
        # Get test suites
        suites = self.get_test_suites()
        
        if not suites:
            print("No test suites selected!")
            return False
        
        print(f"\nRunning {len(suites)} test suite(s)...")
        
        # Run each suite
        for name, suite in suites.items():
            try:
                result = self.run_test_suite(name, suite)
                all_results.append(result)
            except Exception as e:
                print(f"Critical error running {name}: {e}")
                traceback.print_exc()
                all_results.append(None)
        
        # Generate final report
        self.generate_final_report(cov)
        
        # Determine overall success
        success = all(
            result is not None and 
            len(result.failures) == 0 and 
            len(result.errors) == 0 
            for result in all_results
        )
        
        return success
    
    def generate_final_report(self, cov=None):
        """Generate final test report."""
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        print(f"\n{'=' * 80}")
        print("FINAL TEST REPORT")
        print(f"{'=' * 80}")
        
        # Overall statistics
        total_tests = sum(r['tests_run'] for r in self.test_results.values())
        total_success = sum(r['success'] for r in self.test_results.values())
        total_failures = sum(r['failures'] for r in self.test_results.values())
        total_errors = sum(r['errors'] for r in self.test_results.values())
        total_skipped = sum(r['skipped'] for r in self.test_results.values())
        
        print(f"Overall Statistics:")
        print(f"   Total tests: {total_tests}")
        print(f"   Successes: {total_success}")
        print(f"   Failures: {total_failures}")
        print(f"   Errors: {total_errors}")
        print(f"   Skipped: {total_skipped}")
        print(f"   Success rate: {(total_success/total_tests*100):.1f}%" if total_tests > 0 else "   Success rate: N/A")
        print(f"   Total duration: {total_duration:.2f}s")
        
        # Suite breakdown
        print(f"\nSuite Breakdown:")
        for name, results in self.test_results.items():
            success_rate = (results['success'] / results['tests_run'] * 100) if results['tests_run'] > 0 else 0
            status = "PASS" if results['failures'] == 0 and results['errors'] == 0 else "FAIL"
            print(f"   {status} {name}: {results['success']}/{results['tests_run']} ({success_rate:.1f}%) in {results['duration']:.2f}s")
        
        # Performance summary
        if any('Performance' in name for name in self.test_results.keys()):
            print(f"\nPerformance Summary:")
            perf_results = {k: v for k, v in self.test_results.items() if 'Performance' in k}
            for name, results in perf_results.items():
                print(f"   {name}: {results['tests_run']} benchmarks completed")
        
        # Security summary
        security_results = {k: v for k, v in self.test_results.items() if 'Security' in k}
        if security_results:
            print(f"\nSecurity Test Summary:")
            total_security_tests = sum(r['tests_run'] for r in security_results.values())
            total_security_success = sum(r['success'] for r in security_results.values())
            security_success_rate = (total_security_success / total_security_tests * 100) if total_security_tests > 0 else 0
            
            print(f"   Security tests: {total_security_success}/{total_security_tests} ({security_success_rate:.1f}%)")
            if security_success_rate >= 95:
                print("   SECURITY STATUS: EXCELLENT")
            elif security_success_rate >= 85:
                print("   SECURITY STATUS: GOOD")
            else:
                print("   SECURITY STATUS: NEEDS ATTENTION")
        
        # Coverage report
        if cov:
            self.generate_coverage_report(cov)
        
        # Save JSON report
        if self.args.report:
            self.save_json_report(total_duration)
        
        # Final verdict
        print(f"\n{'=' * 80}")
        if total_failures == 0 and total_errors == 0:
            print("ALL TESTS PASSED!")
            if total_skipped > 0:
                print(f"   (Note: {total_skipped} tests were skipped)")
        else:
            print("SOME TESTS FAILED!")
            print("   Please review the failures and errors above.")
        print(f"{'=' * 80}")
    
    def generate_coverage_report(self, cov):
        """Generate code coverage report."""
        try:
            cov.stop()
            cov.save()
            
            print(f"\nCode Coverage Report:")
            
            # Console report
            report_stream = StringIO()
            cov.report(file=report_stream, show_missing=True)
            coverage_text = report_stream.getvalue()
            
            # Extract coverage percentage
            lines = coverage_text.split('\n')
            total_line = [line for line in lines if 'TOTAL' in line]
            if total_line:
                parts = total_line[0].split()
                if len(parts) >= 4:
                    coverage_percent = parts[3].rstrip('%')
                    print(f"   Overall coverage: {coverage_percent}%")
            
            if self.args.verbose:
                print("   Detailed coverage:")
                for line in lines[:10]:  # Show first 10 lines
                    if line.strip():
                        print(f"     {line}")
            
            # HTML report
            if self.args.report:
                html_dir = project_root / 'coverage_html'
                cov.html_report(directory=str(html_dir))
                print(f"   HTML report saved to: {html_dir}")
                
        except Exception as e:
            print(f"   Error generating coverage report: {e}")
    
    def save_json_report(self, total_duration):
        """Save detailed JSON report."""
        try:
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'total_duration': total_duration,
                'python_version': sys.version,
                'project_root': str(project_root),
                'test_results': self.test_results,
                'summary': {
                    'total_tests': sum(r['tests_run'] for r in self.test_results.values()),
                    'total_success': sum(r['success'] for r in self.test_results.values()),
                    'total_failures': sum(r['failures'] for r in self.test_results.values()),
                    'total_errors': sum(r['errors'] for r in self.test_results.values()),
                    'total_skipped': sum(r['skipped'] for r in self.test_results.values())
                }
            }
            
            report_file = project_root / f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            print(f"   Detailed JSON report saved to: {report_file}")
            
        except Exception as e:
            print(f"   Error saving JSON report: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TrafficSignDetector-TensorFlow Security Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_runner.py                    # Run all tests
  python test_runner.py --fast             # Run only fast tests
  python test_runner.py --security         # Run only security tests
  python test_runner.py --performance      # Run only performance tests
  python test_runner.py --coverage         # Run with coverage analysis
  python test_runner.py --verbose --report # Verbose output with detailed report
        """
    )
    
    parser.add_argument('--fast', action='store_true',
                       help='Run only fast tests (skip performance/stress tests)')
    parser.add_argument('--security', action='store_true',
                       help='Run only security tests')
    parser.add_argument('--performance', action='store_true',
                       help='Run only performance tests')
    parser.add_argument('--coverage', action='store_true',
                       help='Generate coverage report (requires coverage.py)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed HTML/JSON reports')
    parser.add_argument('--parallel', action='store_true',
                       help='Run tests in parallel (experimental)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not COVERAGE_AVAILABLE and args.coverage:
        print("Warning: Coverage requested but coverage.py not installed")
        print("Install with: pip install coverage")
        args.coverage = False
    
    # Run tests
    runner = TestSuiteRunner(args)
    success = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()