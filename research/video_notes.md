# Video notes

All 8 videos were processed with the watch skill's **local** engine (YouTube captions + keyframes).
The Gemini engine failed every time (HTTP 503 overload on gemini-3.7-flash and gemini-3.8-flash; 404 on gemini-2.5-flash), see `DECISIONS.md` D3.
Timestamps come from the caption track (MM:SS). "shown" = I saw it in a keyframe; "said" = only in the captions.
Anything I could not confirm from captions or frames is marked [unverified].

## 1. How to label images and audio using Label Studio for object detection and audio classification
Uploader: vision architecture, 13:32. https://www.youtube.com/watch?v=Ctv8FwhkWYA

| Technique / tool | Time | Shown or said |
|---|---|---|
| Label Studio (HumanSignal), installed with pip, run locally on localhost | 00:02, 03:24, 03:59 | shown (login page at 00:00) |
| Anaconda / virtual env for install | 01:04-02:27 | said |
| Bounding-box labeling template ("object detection with bounding boxes"), class names, drag a box, edit/delete | 06:03-07:11 | said [frames for this part not inspected] |
| Export formats: JSON, CSV, COCO, Pascal VOC XML, YOLO txt | 07:56-08:50 | said |
| Next step "train YOLO on the custom data set" | 08:50 | said (not shown in this video) |
| Audio intent classification labeling | 09:31-12:55 | said (not relevant to ARGOS) |

## 2. How to Train YOLO Object Detection Models in Google Colab (YOLO26, YOLO11, YOLOv8)
Uploader: Edje Electronics, 21:25. https://www.youtube.com/watch?v=r0RspiLG260

| Technique / tool | Time | Shown or said |
|---|---|---|
| Build a data set from your own photos in varied conditions; Roboflow Universe, Kaggle, Open Images V7 as sources | 01:17 | said |
| Label Studio labeling, class hotkeys, YOLO export | 02:28-07:31 | said |
| Colab T4 GPU, Ultralytics install, data.yaml config | 08:48-11:07 | said |
| Model families: YOLO26, YOLO11, YOLOv8, YOLOv5; sizes nano..xlarge; pick the smallest model that meets your speed need | 12:11 | said |
| Train yolo11s, 60 epochs, imgsz 640; results.png (loss, precision, recall, mAP) | 13:24-14:36 | said |
| Run locally with a script on a USB camera (`yolo_detect.py --source usb0 --resolution 1280x720`) | 18:58 | said; download step shown at 12:39 frame |

## 3. 3D Eye Tracking on a Webcam
Uploader: Jason Orlosky, 02:38. https://www.youtube.com/watch?v=6J6Rb7f-Cls

| Technique / tool | Time | Shown or said |
|---|---|---|
| Gaze recipe: eye-sphere centres + pupil/iris centres, a ray per eye, intersect rays | 00:11-00:34 | said |
| OpenCV cascade classifier for eyes (mentioned as not enough, no 3D) | 00:46 | said |
| MediaPipe face landmarks (dense face mesh) from one camera image | 00:46-00:59 | said; mesh + eye circles shown at 00:53 frame |
| MediaPipe eye points are biased (shift with eye roll), so anchor to stable points (nose) and build a face coordinate frame / rotation matrix | 01:10-01:34 | said; face box + axes shown at 00:53 |
| MediaPipe iris centres usable without infrared | 01:46 | said |
| Multi-point monitor calibration (not finished in the video) | 02:21 | said |

## 4. YOLOv8: Real-Time Object Detection with Webcam
Uploader: Nicolai Nielsen, 15:22. https://www.youtube.com/watch?v=IHbJcOex6dk

| Technique / tool | Time | Shown or said |
|---|---|---|
| Ultralytics YOLOv8, sizes n/s/m/l/x, speed vs accuracy trade-off (small is his default) | 02:56-04:56 | said |
| `yolo` CLI and Python `model.predict(source=0, show=True)` on a webcam | 05:55, 08:58-11:58 | said; webcam demo frames 12:00-14:23 |
| Export to ONNX, OpenVINO, TensorRT, CoreML, TF for faster deployment; ONNX Runtime / OpenCV DNN inference | 00:57, 06:53, 08:01 | said |
| ~5 ms (200 FPS) on an RTX GPU with the small model | 11:58 | said (GPU numbers, not comparable to our CPU) |

## 5. YOLO Body Tracking on the Pi 5 | Beginner Python Guide
Uploader: Core Electronics, 13:19. https://www.youtube.com/watch?v=KxYcAfuyDoc

| Technique / tool | Time | Shown or said |
|---|---|---|
| YOLO11 pose (17 keypoints), sizes n..x, nano good enough for most uses | 02:56-04:01 | said; keypoint demo frames |
| Export to NCNN (CPU-optimised format) = about 4x FPS | 05:10-06:13 | said |
| Lower processing resolution 640 -> 320 = about 4x FPS again, but loses accuracy and **detection distance** | 06:13-08:27 | said |
| imgsz must be a multiple of 32; use the same imgsz at export and at run time | 08:27 | said |
| Read a keypoint (nose = 0) x/y to drive logic | 09:34 | said |

## 6. YOLOv8 Object Tracking in Real-time with Webcam and OpenCV
Uploader: Nicolai Nielsen, 14:47. https://www.youtube.com/watch?v=UYLp0-iOvFc

| Technique / tool | Time | Shown or said |
|---|---|---|
| SORT tracker on top of YOLOv8 detections (DeepSORT mentioned as an alternative) | 00:10, 05:47 | said |
| Tracker keeps a track through missed detections using its motion model (predict / update) | 04:53-05:47, 09:42 | said |
| SORT parameters: max_age (frames a lost track is kept, he uses 5), min_hits, IoU threshold | 06:44-07:43 | said |
| Draw track centres / trails, colour per ID | 08:43, 12:37 | said; webcam tracking frames 12:43-13:59 |

## 7. YOLOv8: How to Train Object Detection Model with Custom Dataset
Uploader: Nicolai Nielsen, 19:21. https://www.youtube.com/watch?v=TRMZCsBfX78

| Technique / tool | Time | Shown or said |
|---|---|---|
| Roboflow data set + YOLOv8 export format | 00:00, 03:51-04:52 | said |
| Train in Colab (epochs, imgsz, SGD lr 0.01, augmentations), mAP50 / mAP50-95 | 05:51-08:17 | said |
| Under-represented classes hurt (confusion matrix, Roboflow health check) | 09:22 | said |
| Use best.pt, validate, then predict on webcam | 11:20-16:22 | said |

## 8. How to Run Multi-Object Tracking with Ultralytics YOLO26 | BoT-SORT & ByteTrack
Uploader: Ultralytics, 07:46. https://www.youtube.com/watch?v=qQkzKISt5GE

| Technique / tool | Time | Shown or said |
|---|---|---|
| `model.track()` with built-in BoT-SORT (default) and ByteTrack, ID per object | 01:01 | said |
| Tracker YAML: high-confidence threshold to start a track, track_buffer (frames to keep a lost track), match threshold, proximity threshold for re-identification | 02:05 | said [YAML screen not in my sampled frames] |
| Track history trails, line-crossing counting | 03:08-05:08 | said |
| YOLO26n tracking on a MacBook CPU at ~26-30 ms per frame (384x640 input) | 06:13 | said; terminal with "384x640 ... 26.0ms inference" shown in a frame |
| Segmentation masks can be tracked the same way | 07:21 | said |
