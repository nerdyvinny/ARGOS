# Tests for the alert pictures (app/main.py): the saved picture must show the MOMENT of the alert.
# Run from the repo root:  python -m unittest discover -s tests -v

import unittest
import numpy as np
from app import main as app


def entry(t, conf, box=(100, 100, 140, 160)):
    frame = np.full((1080, 1920, 3), int(conf * 100), dtype=np.uint8)
    items = [{"label": "device", "kind": "phone", "box": list(box), "conf": conf}] if conf > 0 else []
    return {"t": t, "frame": frame, "people": [{"box": [0, 0, 400, 600], "conf": 0.9, "points": []}], "items": items}


class EvidencePictureTests(unittest.TestCase):

    def test_device_alert_uses_the_ai_picture_where_the_device_was_clearest(self):
        history = [entry(1.0, 0.5), entry(2.0, 0.9), entry(3.0, 0.0)]   # phone put away at 3.0
        alert = {"kind": "device", "person": "Person 1", "time": 3.2}
        pick = app.pick_evidence(history, alert, [0, 0, 400, 600])
        self.assertEqual(pick["entry"]["t"], 2.0)
        self.assertEqual(pick["item"]["conf"], 0.9)

    def test_device_far_from_the_person_is_not_used(self):
        history = [entry(2.0, 0.9, box=(1500, 900, 1540, 960)), entry(3.0, 0.4)]
        alert = {"kind": "device", "person": "Person 1", "time": 3.2}
        pick = app.pick_evidence(history, alert, [0, 0, 400, 600])
        self.assertEqual(pick["entry"]["t"], 3.0)

    def test_look_alert_uses_the_newest_ai_picture(self):
        history = [entry(1.0, 0.0), entry(2.0, 0.0)]
        alert = {"kind": "long_look", "person": "Person 1", "time": 2.1}
        self.assertEqual(app.pick_evidence(history, alert, [0, 0, 400, 600])["entry"]["t"], 2.0)

    def test_evidence_picture_has_room_and_zoom_side_by_side(self):
        pick = app.pick_evidence([entry(2.0, 0.9)], {"kind": "device", "person": "Person 1", "time": 2.1},
                                 [0, 0, 400, 600])
        img = app.evidence_picture(pick, [0, 0, 400, 600], "Person 1 - phone 00:12")
        self.assertEqual(img.shape[0], 720)          # room picture shrunk to 1280x720
        self.assertGreater(img.shape[1], 1280)       # plus the zoom next to it


if __name__ == "__main__":
    unittest.main()
