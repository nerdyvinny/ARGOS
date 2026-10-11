# Tests for rules/brain.py (the cheating rules).
# Run from the repo root:  python -m unittest discover -s tests -v

import unittest
from rules import brain

STEP = 0.5   # about 2 pictures a second, like the live camera


def face_points(yaw, nose_conf=0.9):
    # 17 body points. Eyes at x 90/110, ears at x 80/120 -> center 100, width 40.
    # The nose is moved sideways to make the head look turned by "yaw".
    points = [[100 + yaw * 40, 50, nose_conf], [90, 45, 0.9], [110, 45, 0.9], [80, 50, 0.9], [120, 50, 0.9]]
    return points + [[0, 0, 0]] * 12


def person(yaw, nose_conf=0.9):
    return {"box": [0, 0, 200, 200], "conf": 0.9, "points": face_points(yaw, nose_conf)}


class Clock:
    # feeds one person a list of pictures, half a second apart
    def __init__(self, state):
        self.state = state
        self.now = 0.0

    def run(self, seconds, yaw, nose_conf=0.9):
        alerts = []
        for i in range(int(seconds / STEP)):
            alerts = alerts + brain.update_person(self.state, person(yaw, nose_conf), False, self.now)
            self.now = self.now + STEP
        return alerts


def kinds(alerts):
    return [a["kind"] for a in alerts]


class HeadTurnTests(unittest.TestCase):

    # B1
    def test_head_down_gap_does_not_make_a_false_long_stare(self):
        s = brain.new_person_state("Person 1", [0, 0, 200, 200])
        clock = Clock(s)
        clock.run(10, 0.0)                     # looks straight: learns the normal angle
        clock.run(1.5, 0.6)                    # short glance (1.5 s, less than a "look away")
        alerts = clock.run(20, 0.0, nose_conf=0.1)   # 20 s head down writing: nose not visible
        alerts = alerts + clock.run(1, 0.0)    # looks up, straight ahead
        self.assertNotIn("long_look", kinds(alerts))
        self.assertEqual(s["total_looks"], 0)

    # B1 (guard: the fix must not break real stares when a face point flickers)
    def test_long_stare_with_flickering_nose_still_alerts(self):
        s = brain.new_person_state("Person 1", [0, 0, 200, 200])
        clock = Clock(s)
        clock.run(10, 0.0)
        alerts = []
        for i in range(16):                    # 8 s stare, nose lost in every 4th picture
            nose_conf = 0.1 if i % 4 == 3 else 0.9
            alerts = alerts + clock.run(STEP, 0.6, nose_conf)
        self.assertIn("long_look", kinds(alerts))

    # B6
    def test_long_stare_during_cooldown_alerts_once_cooldown_ends(self):
        s = brain.new_person_state("Person 1", [0, 0, 200, 200])
        clock = Clock(s)
        clock.run(100, 0.0)                    # long straight history, so the "normal" angle can't drift
        first = clock.run(8, 0.6)              # stare 1 -> alert
        clock.run(4, 0.0)                      # looks back
        second = clock.run(40, 0.6)            # stare 2 starts ~12 s later (inside the 30 s cooldown), lasts 40 s
        self.assertIn("long_look", kinds(first))
        self.assertIn("long_look", kinds(second))


class PhoneMatchTests(unittest.TestCase):

    # B2
    def test_phone_goes_to_the_closest_person_not_the_first(self):
        states = []
        person_a = {"box": [20, 20, 190, 190], "conf": 0.9, "points": []}
        person_b = {"box": [220, 20, 380, 190], "conf": 0.9, "points": []}
        matches = brain.match_people([person_a, person_b], states, 0.0)
        # phone center (222, 100): inside person B's own box,
        # but also inside person A's slightly bigger "near" box
        phone = {"box": [218, 95, 226, 105], "conf": 0.5}
        self.assertEqual(brain.match_phones_to_people([phone], matches), {"Person 2"})


class FollowPeopleTests(unittest.TestCase):

    def test_same_person_keeps_their_name_when_they_move_a_little(self):
        states = []
        brain.match_people([{"box": [0, 0, 100, 200], "conf": 0.9, "points": []}], states, 0.0)
        matches = brain.match_people([{"box": [10, 5, 110, 205], "conf": 0.9, "points": []}], states, 0.5)
        self.assertEqual(list(matches.keys()), ["Person 1"])
        self.assertEqual(len(states), 1)

    def test_new_person_gets_a_new_name(self):
        states = []
        brain.match_people([{"box": [0, 0, 100, 200], "conf": 0.9, "points": []}], states, 0.0)
        matches = brain.match_people([{"box": [0, 0, 100, 200], "conf": 0.9, "points": []},
                                      {"box": [300, 0, 400, 200], "conf": 0.9, "points": []}], states, 0.5)
        self.assertEqual(sorted(matches.keys()), ["Person 1", "Person 2"])


if __name__ == "__main__":
    unittest.main()
