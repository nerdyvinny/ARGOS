# Exam Watch - 360 camera cheating detector

One wide / 360 camera watches the whole test room.
It gives every desk a color and a score, and sends an alert (with a picture) when someone:

1. **has a phone out** (phone seen at their desk for a moment, not a single flicker)
2. **keeps looking at other desks** (turns their head away from *their own normal* for 2+ seconds, 4 times in 5 minutes)
3. **stares at another desk for a long time** (6+ seconds in one go)

The proctor sees everything on a web page: `http://localhost:5000`

> It only *flags* things. A human proctor always makes the final call.

---

## Setup (Windows cmd)

```
cd cheat_cam
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Got an NVIDIA GPU? Install the GPU version of PyTorch first (pick the command at pytorch.org > Get Started), then run the pip line above. Without a GPU it still works, just slower.

## Run it

```
python setup_desks.py
python main.py
```

- `setup_desks.py` takes one picture and lets you draw a box on every desk (ENTER after each box, ESC when done). Do this once after the camera is in place. If the camera moves, do it again.
- `main.py` starts watching. Press **Q** in the video window to stop.
- Practice with a video file instead of the camera: `python setup_desks.py practice.mp4` then `python main.py practice.mp4`
- We use the wide USB camera (Arducam), which is camera `1`. Camera `0` is the laptop camera, and the code refuses to use it: it checks that the picture is 1920 wide, and the laptop camera can only do 1280.
- USB camera not found, or the picture is black? Close the Windows Camera app (only one program can use a camera), then unplug the USB camera, plug it back in, and try again. Never switch the Arducam to MJPG: it goes black until it is replugged. If the numbers changed, try `python main.py 2`.

Output:
- `alerts.csv` - every alert with time, desk and message
- `evidence\` - a picture of the desk for every alert
- web page at `http://localhost:5000` (teammates on the same Wi-Fi can use your laptop's IP)

## Files

| File | What it does |
|---|---|
| `config.py` | All the numbers you can tune |
| `setup_desks.py` | Draw the desk boxes, saves `desks.json` |
| `detector.py` | Runs YOLO: finds people + face points, and phones. Cuts the 360 picture into tiles |
| `brain.py` | Decides who is cheating (head turn math, phone timer, scores) |
| `dashboard.py` | The web page |
| `main.py` | Puts it all together |

---

## How we decided (arguing with ourselves)

**1. How do we know who is who? Face recognition?**
- *For:* sounds cool, judges like it.
- *Against:* you have to sign up every student's face first, it fails when heads are down writing, and storing student faces is a privacy problem.
- **Decision: the desk IS the ID.** Draw the desks once. Whoever sits in Desk 4 is "Desk 4". No tracking, no face database. (Stretch goal: match a desk to a seating chart name.)

**2. How do we catch phones?**
- *Idea A:* "head down for a long time" = phone. *Problem:* writing on the test also means head down. Tons of false alarms.
- *Idea B:* YOLO already knows "cell phone" (COCO class 67).
- **Decision: B.** Phone must show up 3 times in 4 seconds (we only get about 2 pictures a second on a laptop) so one random flicker doesn't count. We lowered the confidence to 0.25 because phones are small.

**3. How do we know someone looks at another desk?**
- *Idea A:* MediaPipe face mesh for exact head angle. *Problem:* from across the room faces are tiny, face mesh needs a big face.
- *Idea B:* YOLO pose gives nose, eyes and ears. If the nose moves off-center between the eyes/ears, the head is turned.
- **Decision: B**, it comes free with the person detector.
- *New problem:* the camera sees every desk from a different side, so "straight ahead" looks different for each person.
- **Fix: each desk learns its own normal** (median head angle over the last 2 minutes). We only count turning *away from your own normal*.
- *Another problem:* people stretch, look at the clock, crack their neck.
- **Fix: time + count.** Short glances (< 2 s) don't count. 4 real look-aways in 5 minutes = alert. One very long stare (6 s) = alert right away.

**4. Where do we put the 360 camera?**
- *Middle of the room:* sees everyone, but half the class shows the back of their head, so no nose = no head angle.
- *Front of the room, up high:* sees almost every face. A 360 cam only needs to use the front half.
- **Decision: front, up high (on the teacher's desk or a tripod ~2 m tall).** Point the 360 "seam" at the wall so nobody gets cut in half.

**5. The 360 picture is huge and people look tiny.**
- *Just shrink it:* people become a few pixels, AI misses them.
- **Decision: cut off ceiling and floor, then cut it into 3 tiles with a little overlap.** Run YOLO on each tile, then remove the people that show up twice in the overlap.

**6. Do we need 30 fps?** No. Cheating takes seconds. 3-10 fps is plenty, so a laptop is fine. For video files we use the video's own clock, so a slow laptop still measures seconds right.

---

## Team split (4 people)

| Person | Job |
|---|---|
| 1 - Camera | Get the 360 cam working as a webcam (USB webcam mode, panorama view), place it, record practice videos of the team "cheating" and "not cheating" |
| 2 - AI | `detector.py`: test YOLO on the practice videos, tune `IMAGE_SIZE`, `TILES`, confidences. Stretch: fine-tune YOLO on "phone in hand / under desk" photos |
| 3 - Brain | `brain.py` + `config.py`: tune the thresholds on the practice videos until honest people stay green and cheaters go red |
| 4 - Demo | `dashboard.py`: make it look great, write the pitch, slides, run the live demo |

## 36 hour timeline

| Hours | Goal |
|---|---|
| 0-3 | Everyone installs, run the code on a laptop webcam. Camera person gets the 360 cam streaming |
| 3-8 | Record 3-4 practice videos (normal test, phone cheater, looker, mix). Draw desks on them |
| 8-16 | Tune detection + thresholds on the videos. Dashboard polish starts |
| 16-20 | Sleep in shifts! |
| 20-28 | Live test in a real room with the real camera. Fix what breaks |
| 28-32 | Stretch goals (seating chart names, sound alert, fine-tuned phone model) |
| 32-36 | **Freeze the code.** Practice the demo 3 times. Have a backup video in case the live camera fails |

## Tuning cheat sheet (`config.py`)

- Too many false "looking" alerts -> raise `YAW_THRESHOLD` (0.25 -> 0.35) or `LOOK_AWAY_SECONDS`
- Missing real lookers -> lower `YAW_THRESHOLD`, lower `LOOK_COUNT_LIMIT`
- Phones not found -> lower `PHONE_CONFIDENCE`, raise `IMAGE_SIZE` to 1280
- Too slow -> `TILES = 2`, `IMAGE_SIZE = 640`
- People not found -> raise `TILES`, check `CROP_TOP` / `CROP_BOTTOM` aren't cutting people off

## Known limits (say these to the judges, it shows you thought about it)

- A phone hidden under the desk can't be seen by any camera. Head-down + hands-under-desk could be a stretch "weak signal".
- Students facing away from the camera have no face points, so head turn can't be measured for them.
- It flags, a human decides. Evidence pictures should be deleted after the test.


