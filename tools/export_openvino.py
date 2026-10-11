# Makes the faster OpenVINO copies of the AI models (run once per laptop, after the .pt files exist).
# The detector uses a "<name>_openvino_model" folder automatically when it is there,
# and falls back to the normal .pt file when it is not.
#
#   python tools\export_openvino.py            (models next to the code)
#   python tools\export_openvino.py models     (models in another folder)
#
# YOLO-World's words are baked into the OpenVINO copy, so run this again after changing WORLD_WORDS.

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from settings import config
from vision import detector


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    for name in [config.POSE_MODEL, config.OBJECT_MODEL]:
        model = YOLO(os.path.join(folder, os.path.basename(name)))
        model.export(format="openvino", imgsz=config.IMAGE_SIZE, dynamic=True, half=False)
    world = YOLO(os.path.join(folder, os.path.basename(detector.WORLD_MODEL)))
    world.set_classes(list(detector.WORLD_WORDS.keys()))
    world.export(format="openvino", imgsz=config.IMAGE_SIZE, dynamic=True, half=False)
    print("Done. The detector will now use the OpenVINO copies.")


if __name__ == "__main__":
    main()
