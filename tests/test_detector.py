# Tests for vision/detector.py. No AI model is loaded: the model is replaced by a fake.
# Run from the repo root:  python -m unittest discover -s tests -v

import unittest
import numpy as np
from vision import detector


class FakeArray:
    # looks like a torch tensor: .cpu().numpy()
    def __init__(self, values):
        self.values = np.array(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    def __init__(self, boxes):
        self.xyxy = FakeArray(boxes)
        self.conf = FakeArray([0.9] * len(boxes))

    def __len__(self):
        return len(self.xyxy.values)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = FakeBoxes(boxes)
        self.keypoints = None


class FakeModel:
    # gives back the boxes we planned for each tile (in that tile's own numbers)
    def __init__(self, boxes_per_tile):
        self.boxes_per_tile = boxes_per_tile

    def __call__(self, images, **kwargs):
        return [FakeResult(b) for b in self.boxes_per_tile]


class RemoveDoublesTests(unittest.TestCase):

    # B3
    def test_two_real_people_in_one_tile_are_both_kept(self):
        saved = detector.pose_model
        try:
            detector.pose_model = FakeModel([
                # tile 0 (starts at x=0): a front-row student, a back-row student mostly
                # hidden behind them (64% of the smaller box overlaps), and a student at the tile edge
                [[100, 500, 400, 1000], [180, 420, 320, 640], [500, 0, 700, 300]],
                # tile 1 (starts at x=544): the same edge student again, cut off by the tile
                [[0, 0, 156, 300]],
            ])
            frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
            people = detector.find_people([(0, 0, frame), (544, 0, frame)])
        finally:
            detector.pose_model = saved
        boxes = sorted([[int(v) for v in p["box"]] for p in people])
        # front + back student kept, the edge student counted once
        self.assertEqual(boxes, [[100, 500, 400, 1000], [180, 420, 320, 640], [500, 0, 700, 300]])


class FakeWorldBoxes(FakeBoxes):
    def __init__(self, boxes, classes):
        FakeBoxes.__init__(self, boxes)
        self.cls = FakeArray(classes)


class FakeWorldResult:
    def __init__(self, boxes, classes):
        self.boxes = FakeWorldBoxes(boxes, classes)
        self.keypoints = None


class FakeWorldModel:
    # tile 0: finds a smartwatch (class 0) and a name tag (class 3); close-ups find nothing
    def __call__(self, images, **kwargs):
        out = [FakeWorldResult([[10, 10, 50, 50], [300, 300, 360, 340]], [0, 3])]
        for i in range(len(images) - 1):
            out.append(FakeWorldResult([], []))
        return out


class DeviceTests(unittest.TestCase):

    def test_phone_next_to_headphones_is_kept(self):
        things = [
            {"box": [0, 0, 200, 200], "label": "headphones", "tile": 0},
            {"box": [50, 50, 100, 120], "label": "phone", "tile": 0},
        ]
        labels = sorted(t["label"] for t in detector.remove_doubles_per_label(things))
        self.assertEqual(labels, ["headphones", "phone"])

    def test_name_tag_is_not_a_phone_but_watch_is_found(self):
        saved = (detector.pose_model, detector.object_model, detector.world_model)
        try:
            detector.pose_model = FakeModel([[]])
            detector.object_model = FakeModel([[[302, 302, 358, 338]]])   # "phone" that is really the name tag
            detector.world_model = FakeWorldModel()
            frame = np.zeros((1000, 1000, 3), dtype=np.uint8)
            saved_tiles = detector.config.TILES
            detector.config.TILES = 1
            people, phones = detector.detect(frame)
            detector.config.TILES = saved_tiles
        finally:
            detector.pose_model, detector.object_model, detector.world_model = saved
        self.assertEqual([p["label"] for p in phones], ["smartwatch"])


if __name__ == "__main__":
    unittest.main()
