# -*- coding: utf-8 -*-
import os
import sys
import unittest
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


class DualOutput:
    """Tee output to both stdout and a file."""
    def __init__(self, file_path):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def main():
    log_file = SCRIPT_DIR / "tests" / "test_results.log"
    dual_out = DualOutput(log_file)
    sys.stdout = dual_out
    sys.stderr = dual_out

    print(f"======================================================================")
    print(f" Franka Obstacle Avoidance & Perception - Automated Test Suite")
    print(f" Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Log File: {log_file}")
    print(f"======================================================================\n")

    loader = unittest.TestLoader()
    suite = loader.discover(str(SCRIPT_DIR / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(suite)

    print(f"\n======================================================================")
    print(f" Summary: {result.testsRun} tests run.")
    print(f" Status: {'ALL PASSED (SUCCESS)' if result.wasSuccessful() else 'FAILED'}")
    print(f" Failures: {len(result.failures)}, Errors: {len(result.errors)}")
    print(f"======================================================================")

    dual_out.close()
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
