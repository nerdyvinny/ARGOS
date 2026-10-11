# Tests for vision/tracker.py (session numbers, lost pool, merges, smoothing).
# Run from the repo root:  python -m unittest discover -s tests -v

import unittest
import numpy as np
from vision import tracker as tr


def frame():
    # grey picture with a blue "shirt" block, so clothes colours can be measured
    f = np.full((480, 640, 3), 128, dtype=np.uint8)
    f[150:300, 250:390] = (200, 60, 30)
    return f


def det(box, conf=0.9):
    return {"box": list(box), "conf": conf, "points": []}


PERSON = [240, 100, 400, 460]


def feed(t, dets, now):
    t.predict(frame(), now)
    t.update(dets, now)


def moving_frame(x):
    # plain picture with a speckled "person" whose left edge is at x (optical flow needs texture)
    f = np.full((480, 640, 3), 90, dtype=np.uint8)
    rng = np.random.default_rng(7)
    f[100:460, x:x + 160] = rng.integers(0, 255, (360, 160, 3), dtype=np.uint8)
    return f


class TrackerTests(unittest.TestCase):

    def test_old_ai_answer_is_moved_forward_to_where_the_person_is_now(self):
        # The AI answers for an older picture. The box must end up where the person is NOW,
        # not jump back to where they were when the AI looked (the outline lagged behind).
        t = tr.Tracker(0.4)
        for i in range(3):
            t.predict(moving_frame(100), i * 0.03, index=i)
            t.update([det([100, 100, 260, 460])], i * 0.03, index=i)
        x = 100
        for i in range(3, 9):                 # walks right 10 px per picture
            x = x + 10
            t.predict(moving_frame(x), i * 0.03, index=i)
        # AI result for picture 4 (person at 120) arrives while picture 8 (person at 160) is on screen
        t.update([det([120, 100, 280, 460])], 9 * 0.03, index=4)
        box = t.current()[0]["box"]
        self.assertAlmostEqual(box[0], 160, delta=4)

    def test_new_person_needs_three_sightings_before_a_number(self):
        t = tr.Tracker(0.4)
        feed(t, [det(PERSON)], 0.0)
        feed(t, [det(PERSON)], 0.5)
        self.assertEqual(t.current(), [])
        feed(t, [det(PERSON)], 1.0)
        self.assertEqual([p["track_id"] for p in t.current()], [1])

    def test_unsure_detection_keeps_a_person_but_never_starts_one(self):
        t = tr.Tracker(0.4)
        for i in range(3):
            feed(t, [det(PERSON)], i * 0.5)
        feed(t, [det(PERSON, conf=0.3), det([10, 10, 60, 120], conf=0.3)], 1.5)
        self.assertEqual([p["track_id"] for p in t.current()], [1])
        self.assertTrue(t.current()[0]["fresh"])

    def test_person_who_leaves_and_comes_back_gets_the_same_number(self):
        t = tr.Tracker(0.4)
        for i in range(3):
            feed(t, [det(PERSON)], i * 0.5)
        for i in range(4):                       # gone for a while (misses -> lost pool)
            feed(t, [], 2.0 + i)
        self.assertEqual(t.current(), [])
        feed(t, [det([250, 105, 410, 465])], 20.0)   # back 18 s later, same place and clothes
        self.assertEqual([p["track_id"] for p in t.current()], [1])

    def test_two_numbers_on_one_person_merge_and_keep_the_older(self):
        t = tr.Tracker(0.4)
        for i in range(3):
            feed(t, [det(PERSON)], i * 0.5)
        far = [500, 100, 600, 300]
        for i in range(3):
            feed(t, [det(PERSON), det(far)], 1.5 + i * 0.5)
        self.assertEqual(sorted(p["track_id"] for p in t.current()), [1, 2])
        t.active[1]["box"] = [242, 102, 402, 462]   # number 2 turns out to sit on person 1
        for i in range(tr.MERGE_UPDATES):
            t.merge_doubles(3.0 + i)
        self.assertIn((1, 2), t.take_merges())
        self.assertEqual([p["track_id"] for p in t.current()], [1])

    def test_reset_forgets_everyone(self):
        t = tr.Tracker(0.4)
        for i in range(3):
            feed(t, [det(PERSON)], i * 0.5)
        t.reset()
        self.assertEqual(t.current(), [])
        self.assertEqual(t.lost, [])
        self.assertEqual(t.next_number, 1)

    def test_one_euro_filter_steadies_small_shakes(self):
        f = tr.OneEuro()
        raw = [100 + (2 if i % 2 else -2) for i in range(30)]   # shaking +-2 px around 100
        out = [f(x, i * 0.25) for i, x in enumerate(raw)]
        self.assertLess(np.std(out[10:]), np.std(raw[10:]) * 0.6)


if __name__ == "__main__":
    unittest.main()
