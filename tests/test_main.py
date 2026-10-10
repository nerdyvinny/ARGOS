# Tests for app/main.py. The camera/video, the AI models and the window are all faked,
# and every file the app writes goes into a temporary folder.
# Run from the repo root:  python -m unittest discover -s tests -v

import os
import io
import sys
import json
import tempfile
import contextlib
import unittest
from unittest import mock
import numpy as np
from settings import config
from capture import camera
from rules import brain
from vision import detector
from app import main as app_main

W = 64
H = 48


class FakeVideo:
    # a pretend video file: every picture's pixels hold its picture number
    def __init__(self, frames, fps=30.0):
        self.frames = frames
        self.fps = fps
        self.pos = 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, prop):
        return self.fps

    def set(self, prop, value):
        return True

    def grab(self):
        if self.pos >= self.frames:
            return False
        self.pos = self.pos + 1
        return True

    def read(self):
        if self.pos >= self.frames:
            return False, None
        frame = np.full((H, W, 3), self.pos % 256, dtype=np.uint8)
        self.pos = self.pos + 1
        return True, frame

    def release(self):
        self.released = True


def picture_number(frame):
    return int(frame[0, 0, 0])


class MainLoopTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        desks_file = os.path.join(self.tmp.name, "desks.json")
        with open(desks_file, "w") as f:
            json.dump({"frame_width": W, "frame_height": H,
                       "desks": [{"name": "Desk 1", "box": [0, 0, W, H]}]}, f)
        self.patches = [
            mock.patch.object(config, "DESKS_FILE", desks_file),
            mock.patch.object(config, "EVIDENCE_FOLDER", os.path.join(self.tmp.name, "evidence")),
            mock.patch.object(config, "ALERTS_CSV", os.path.join(self.tmp.name, "alerts.csv")),
            mock.patch.object(config, "SHOW_WINDOW", False),
            mock.patch.object(config, "DASHBOARD", False),
            mock.patch.object(sys, "argv", ["app.main", "practice.mp4"]),
            mock.patch.object(detector, "load_models", lambda: None),
        ]
        for p in self.patches:
            p.start()
        self.times = []
        self.alerts = []

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def run_main(self, video, detect):
        real_update_all = brain.update_all

        def record(desk_states, people, phones, now):
            self.times.append(now)
            alerts, matches = real_update_all(desk_states, people, phones, now)
            self.alerts = self.alerts + alerts
            return alerts, matches

        output = io.StringIO()
        with mock.patch.object(camera, "open_camera", lambda source: video), \
                mock.patch.object(detector, "detect", detect), \
                mock.patch.object(brain, "update_all", record), \
                contextlib.redirect_stdout(output):
            app_main.main()
        return output.getvalue()

    # B5
    def test_video_file_is_checked_at_live_camera_speed(self):
        self.run_main(FakeVideo(90), lambda frame: ([], []))     # 3 seconds of a 30 fps video
        gaps = [round(b - a, 3) for a, b in zip(self.times, self.times[1:])]
        # the live camera gives about 2 pictures a second -> half a second between checks
        self.assertEqual(set(gaps), {0.5})

    # B5
    def test_phone_seen_for_a_tenth_of_a_second_in_a_video_does_not_alert(self):
        def detect(frame):
            if picture_number(frame) in (30, 31, 32):           # 3 pictures = 0.1 s at 30 fps
                return [], [{"box": [10, 10, 20, 20], "conf": 0.9, "tile": 0}]
            return [], []
        self.run_main(FakeVideo(60), detect)
        self.assertNotIn("phone", [a["kind"] for a in self.alerts])

    # B10
    def test_camera_released_when_first_picture_cannot_be_read(self):
        video = FakeVideo(0)                                    # opens, but gives no picture
        self.run_main(video, lambda frame: ([], []))
        self.assertTrue(video.released)

    # B10
    def test_camera_released_when_the_detector_crashes(self):
        video = FakeVideo(90)

        def broken_detect(frame):
            raise RuntimeError("model crashed")
        with self.assertRaises(RuntimeError):
            self.run_main(video, broken_detect)
        self.assertTrue(video.released)

    # B10
    def test_ctrl_c_releases_camera_and_still_prints_summary(self):
        video = FakeVideo(90)
        calls = []

        def detect(frame):
            calls.append(1)
            if len(calls) == 3:
                raise KeyboardInterrupt                         # the proctor presses Ctrl+C
            return [], []
        try:
            output = self.run_main(video, detect)
        except KeyboardInterrupt:
            self.fail("Ctrl+C was not handled: no summary, camera not released")
        self.assertTrue(video.released)
        self.assertIn("SUMMARY", output)

    # B10 (guard: Ctrl+C while the AI models are still loading)
    def test_ctrl_c_while_models_load_releases_camera(self):
        video = FakeVideo(90)

        def interrupted_load():
            raise KeyboardInterrupt
        with mock.patch.object(detector, "load_models", interrupted_load):
            try:
                output = self.run_main(video, lambda frame: ([], []))
            except KeyboardInterrupt:
                self.fail("Ctrl+C was not handled")
        self.assertTrue(video.released)
        self.assertIn("SUMMARY", output)


if __name__ == "__main__":
    unittest.main()
