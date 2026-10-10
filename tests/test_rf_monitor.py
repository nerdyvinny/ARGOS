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
        result = run(["-m", "rf_detect.rf_monitor", "scan", "--simulate", "--no-web"], REPO)
        self.assertEqual(result.returncode, 0, result.stderr)

    # B9 (guard: the old way, from inside the rf_detect folder, still works)
    def test_runs_from_inside_the_rf_detect_folder(self):
        result = run(["rf_monitor.py", "scan", "--simulate", "--no-web"], os.path.join(REPO, "rf_detect"))
        self.assertEqual(result.returncode, 0, result.stderr)

    # no batch file needed: run the file by its path from the repo root
    def test_runs_by_path_from_the_repo_root(self):
        result = run([os.path.join("rf_detect", "rf_monitor.py"), "scan", "--simulate", "--no-web"], REPO)
        self.assertEqual(result.returncode, 0, result.stderr)


class HeatmapTests(unittest.TestCase):

    def test_scan_writes_the_waterfall(self):
        result = run([os.path.join("rf_detect", "rf_monitor.py"), "scan", "--simulate", "--no-web"], REPO)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.path.exists(os.path.join(REPO, "rf_detect", "rf_waterfall.html")))

    def test_survey_finds_the_fake_phone_desk(self):
        result = run([os.path.join("rf_detect", "rf_monitor.py"), "survey", "--simulate", "--no-web"], REPO)
        self.assertEqual(result.returncode, 0, result.stderr)
        # the fake phone sits at the 3rd desk, so it should be the loudest
        loudest = result.stdout.split("Loudest desks:")[1].split()[0:2]
        self.assertEqual(loudest, ["Desk", "3"], result.stdout)


if __name__ == "__main__":
    unittest.main()
