#!/usr/bin/env python3
"""
Test runner script for RAG system.
Sets up proper test environment and runs all tests.
"""
import os
import sys
import subprocess
from pathlib import Path

def setup_test_environment():
    """Set up environment variables for testing"""
    test_env = {
        "OPENAI_API_KEY": "test-api-key-for-testing",
        "MINIO_USERNAME": "testuser", 
        "MINIO_PASSWORD": "testpass",
        "FLASK_DEBUG": "false",
        "LOG_LEVEL": "ERROR",
        "TESTING": "true"
    }
    
    # Update environment
    for key, value in test_env.items():
        os.environ[key] = value
    
    print("🔧 Test environment configured")

def run_tests(test_type="all", verbose=True):
    """Run tests with proper configuration"""
    setup_test_environment()
    
    # Determine which tests to run
    if test_type == "unit":
        test_path = "tests/test_chunking_strategies.py"
    elif test_type == "integration":
        test_path = "tests/test_rag_service_integration.py"
    elif test_type == "api":
        test_path = "tests/test_api_endpoints.py"
    else:
        test_path = "tests/"
    
    # Build pytest command
    cmd = ["python", "-m", "pytest", test_path]
    
    if verbose:
        cmd.append("-v")
    
    # Add coverage if available
    try:
        import coverage
        cmd.extend(["--cov=.", "--cov-report=term-missing"])
    except ImportError:
        pass
    
    print(f"🧪 Running tests: {' '.join(cmd)}")
    print("=" * 60)
    
    # Run tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAG system tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "api"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Run tests in quiet mode"
    )
    
    args = parser.parse_args()
    
    print("🚀 RAG System Test Runner")
    print("=" * 60)
    
    success = run_tests(
        test_type=args.type,
        verbose=not args.quiet
    )
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()