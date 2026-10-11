# Tests for the device filter in vision/detector.py (pens and random things must not become devices).
# Run from the repo root:  python -m unittest discover -s tests -v

import unittest
from vision import detector


def thing(label, box, conf, src="full"):
    return {"label": label, "box": list(box), "conf": conf, "tile": None, "src": src}


PHONE = (100, 100, 160, 220)     # 60 x 120: phone shaped (2 to 1)
PEN = (300, 100, 312, 220)       # 12 x 120: pen shaped (10 to 1)


class DeviceFilterTests(unittest.TestCase):

    def test_sure_phone_is_kept(self):
        kept = detector.filter_devices([thing("phone", PHONE, 0.6)], [])
        self.assertEqual(len(kept), 1)

    def test_unsure_phone_is_dropped(self):
        self.assertEqual(detector.filter_devices([thing("phone", PHONE, 0.3)], []), [])

    def test_close_up_needs_to_be_surer_than_the_whole_picture(self):
        self.assertEqual(detector.filter_devices([thing("phone", PHONE, 0.4, src="closeup")], []), [])
        self.assertEqual(len(detector.filter_devices([thing("phone", PHONE, 0.4, src="full")], [])), 1)

    def test_long_thin_phone_box_is_a_pen_not_a_phone(self):
        self.assertEqual(detector.filter_devices([thing("phone", PEN, 0.6)], []), [])

    def test_phone_where_yolo_world_sees_a_pen_is_dropped(self):
        pen = thing("pen", PHONE, 0.5)
        self.assertEqual(detector.filter_devices([thing("phone", PHONE, 0.55)], [pen]), [])

    def test_phone_much_surer_than_the_pen_is_kept(self):
        pen = thing("pen", PHONE, 0.2)
        self.assertEqual(len(detector.filter_devices([thing("phone", PHONE, 0.8)], [pen])), 1)

    def test_unsure_smartwatch_from_yolo_world_is_dropped(self):
        self.assertEqual(detector.filter_devices([thing("smartwatch", PHONE, 0.2)], []), [])
        self.assertEqual(len(detector.filter_devices([thing("smartwatch", PHONE, 0.5)], [])), 1)

    def test_pen_and_paper_pass_through_untouched(self):
        # they are reported (never alerted on), so the filter leaves them alone
        kept = detector.filter_devices([thing("pen", PEN, 0.2), thing("paper", PHONE, 0.2)], [])
        self.assertEqual(len(kept), 2)


if __name__ == "__main__":
    unittest.main()
