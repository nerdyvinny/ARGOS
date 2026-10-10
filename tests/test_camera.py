# Tests for capture/camera.py. cv2.VideoCapture is always faked: no real camera is ever opened.
# Run from the repo root:  python -m unittest discover -s tests -v

import sys
import unittest
from unittest import mock
import cv2
import numpy as np
from capture import camera


def fake_cap(width, height):
    cap = mock.MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, np.full((height, width, 3), 100, dtype=np.uint8))
    return cap


class OpenCameraTests(unittest.TestCase):

    # B4
    def test_laptop_camera_0_is_never_opened(self):
        with mock.patch("cv2.VideoCapture") as video_capture:
            video_capture.return_value = fake_cap(1920, 1080)
            cap = camera.open_camera(0)
        self.assertIsNone(cap)
        video_capture.assert_not_called()

    # B4 (guard: the USB camera 1 still opens the same way)
    def test_usb_camera_1_still_opens(self):
        with mock.patch("cv2.VideoCapture") as video_capture:
            video_capture.return_value = fake_cap(1920, 1080)
            cap = camera.open_camera(1)
        self.assertIsNotNone(cap)
        if sys.platform == "win32":
            video_capture.assert_called_once_with(1, cv2.CAP_DSHOW)

    # B4 (guard: a small picture is still refused and the camera let go)
    def test_small_picture_is_still_refused(self):
        with mock.patch("cv2.VideoCapture") as video_capture:
            small = fake_cap(1280, 720)
            video_capture.return_value = small
            cap = camera.open_camera(1)
        self.assertIsNone(cap)
        small.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
