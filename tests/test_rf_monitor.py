# Tests for rf_detect/rf_monitor.py. Uses --simulate, so no radio is needed.
# Run from the repo root:  python -m unittest discover -s tests -v

import os
import sys
import subprocess
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(args, folder):
    return subprocess.run([sys.executable] + args, cwd=folder, capture_output=True, text=True, timeout=60)


class ImportTests(unittest.TestCase):

    # B9
    def test_runs_as_a_module_from_the_repo_root(self):
        result = run(["-m", "rf_detect.rf_monitor", "scan", "--simulate"], REPO)
        self.assertEqual(result.returncode, 0, result.stderr)

    # B9 (guard: the old way, from inside the rf_detect folder, still works)
    def test_runs_from_inside_the_rf_detect_folder(self):
        result = run(["rf_monitor.py", "scan", "--simulate"], os.path.join(REPO, "rf_detect"))
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
