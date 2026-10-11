# ARGOS
Anti-Cheating, for teachers.

## Exam Watch - 360 camera cheating detector

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
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Always run the commands below from the repo root (the top folder) (the models, `desks.json`, `alerts.csv` and `evidence\` are found relative to it).

Got an NVIDIA GPU? Install the GPU version of PyTorch first (pick the command at pytorch.org > Get Started), then run the pip line above. Without a GPU it still works, just slower.

## Run it

```
python -m tools.setup_desks
python -m app.main
```

- `tools/setup_desks.py` takes one picture and lets you draw a box on every desk (ENTER after each box, ESC when done). Do this once after the camera is in place. If the camera moves, do it again.
- `app/main.py` starts watching. Press **Q** in the video window to stop.
- Practice with a video file instead of the camera: `python -m tools.setup_desks practice.mp4` then `python -m app.main practice.mp4`
- We use the wide USB camera (Arducam), which is camera `1`. Camera `0` is the laptop camera, and the code refuses to use it: it checks that the picture is 1920 wide, and the laptop camera can only do 1280.
- USB camera not found, or the picture is black? Close the Windows Camera app (only one program can use a camera), then unplug the USB camera, plug it back in, and try again. Never switch the Arducam to MJPG: it goes black until it is replugged. If the numbers changed, try `python -m app.main 2`.

Output:
- `alerts.csv` - every alert with time, desk and message
- `evidence\` - a picture of the desk for every alert
- web page at `http://localhost:5000` (teammates on the same Wi-Fi can use your laptop's IP)

## Files

```
ARGOS/
  app/main.py            puts it all together (start here)
  capture/camera.py      opens the USB camera (or a video file)
  vision/detector.py     YOLO models + 360 tiles
  rules/brain.py         cheating rules and scores
  ui/dashboard.py        the web page
  settings/config.py     all the numbers you can tune
  tools/setup_desks.py   draw the desk boxes, saves desks.json
  rf_detect/             radio detector: phones transmitting + Bluetooth headphones (see below)
```

| File | What it does |
|---|---|
| `settings/config.py` | All the numbers you can tune |
| `tools/setup_desks.py` | Draw the desk boxes, saves `desks.json` |
| `vision/detector.py` | Runs YOLO: finds people + face points, and phones. Cuts the 360 picture into tiles |
| `rules/brain.py` | Decides who is cheating (head turn math, phone timer, scores) |
| `ui/dashboard.py` | The web page |
| `app/main.py` | Puts it all together |

---

## Radio detector: phones + Bluetooth headphones (`rf_detect/`)

The camera can't see a phone in a pocket or under a desk. But a phone that's being used still **sends radio signals**, and so do wireless earbuds. This part listens for them.

It has two parts:

1. **Cell phone detector.** A cheap USB radio (an RTL-SDR) listens to the frequencies phones use to talk *to* the cell tower. When a phone in the room is in use, the radio power on those frequencies goes up.
2. **Bluetooth headphone detector.** The laptop's own Bluetooth looks for headphones and earbuds that are switched on nearby.

> It only measures **how much** signal there is. It never decodes, records or listens to anything, so it can't tell what anyone is doing on their phone.

It runs separately from the camera for now. The link between the two is a placeholder (see [Camera link](#camera-link-todo)).

### Picture of what the radio does

```
   FRONT OF ROOM                                   STUDENTS' DESKS
  +-------------+
  |   CAMERA    |
  +-------------+          ))) ))  )   <- phone sends a signal
         |                 )) )          (up to the cell tower)
     \ | /   ((( (( (  <-- the antenna catches it
      \|/
       Y  antenna  ------>  [ RTL-SDR ]  ------>  [ Laptop ]
       |                    USB radio             measures power
  ============                                    in each band
  metal sheet
  (blocks the back)

  Empty room:  band power  ▁▁▁▁▁▁▁▁   = normal (baseline)
  Phone in use: band power ▁▁▁█▇▁▁▁   = louder than normal -> ACTIVE
```

The antenna is like a radar that only listens: it never sends anything out. It just measures how loud each phone band is and checks whether it's louder than the empty room.

### How it works

**Phones:**
- First you record the **empty room** (the "baseline"), so the program knows what normal looks like.
- Then it sweeps through the phone bands over and over (about 15 seconds per sweep) and compares each one to normal.
- If a band is louder than normal in 2 of the last 5 sweeps, it says **ACTIVE**. One random blip doesn't count.

**Bluetooth:**
- Headphones that are switched on keep sending short "I'm here" messages. It recognizes:
  - **AirPods / Beats**, from Apple's status message. They send it while in use, which is how iPhones show their battery.
  - **Most Android earbuds**, from the Google Fast Pair signal.
  - **New LE Audio** headphones.
  - **Headphone-like names**, such as "AirBuds", "WH-1000XM5", JBL, Bose or Jabra.
- Signal strength gives a rough distance: very close, in the room, or far.

### Bands it listens to

Phones *transmit* on the "uplink" half of each band, so these are uplink frequencies. 4G and 5G use the same frequencies.

| Band | MHz | Carriers |
|---|---|---|
| B71 | 663-698 | T-Mobile, Boost |
| B12/17 | 699-716 | AT&T, T-Mobile, US Cellular |
| B13 | 777-787 | Verizon |
| B14 | 788-798 | AT&T FirstNet |
| B26 | 817-824 | T-Mobile (old Sprint) |
| B5 | 824-849 | AT&T, Verizon, US Cellular |
| B70 | 1695-1710 | Boost (Dish) |
| B66 | 1710-1765 | AT&T, Verizon, T-Mobile |

Smaller carriers rent the big carriers' networks, so they're covered too. Metro, Mint and Google Fi run on T-Mobile; Visible and Straight Talk on Verizon; Cricket on AT&T.

### What you need

- An **RTL-SDR** USB radio with its antenna. Ours has the R820T tuner, which reaches up to about 1766 MHz.
- **SDR++** unzipped somewhere. We borrow its `rtlsdr.dll` driver. The folder is set in `RTLSDR_DLL_FOLDER` in `rf_detect/rf_config.py`; by default it's `Downloads\sdrpp_windows_x64`.
- A laptop with **Bluetooth turned on**: Windows Settings > Bluetooth & devices.

### Setup (Windows cmd)

This part needs **Python 3.12**, so use `py -3.12`. The Bluetooth library doesn't install on Python 3.15 yet.

```
py -3.12 -m pip install bleak numpy pyrtlsdr==0.2.93 "setuptools<81"
```

Stick to `pyrtlsdr==0.2.93`: newer versions don't work with the `rtlsdr.dll` that comes with SDR++.

### Run it

Close SDR++ first. Only one program can use the radio at a time.

```
py -3.12 rf_detect\rf_monitor.py scan
py -3.12 rf_detect\rf_monitor.py baseline
py -3.12 rf_detect\rf_monitor.py fovtest
py -3.12 rf_detect\rf_monitor.py monitor
py -3.12 rf_detect\rf_monitor.py bluetooth
```

1. `scan` - one sweep. A good first test that the radio works.
2. `baseline` - records the **empty room**, with phones off. Redo it if you move the antenna or change rooms.
3. `fovtest` - the range test, which keeps the alarm to this room (see below).
4. `monitor` - watches for phones, and checks Bluetooth in the background. Press **Ctrl+C** to stop.
5. `bluetooth` - Bluetooth only. It doesn't need the RTL-SDR.

Extra options:
- `--simulate` - a fake radio and fake Bluetooth, to try it with no hardware.
- `--sweeps 10` - stop after 10 sweeps.
- `--no-web` - don't open the web page.

The web page opens by itself at `http://localhost:5050` (the camera dashboard is on 5000):
- `/waterfall` - which phone band is busy, over time (red = louder than normal)
- `/bluetooth` - headphones switched on nearby, closest first

Output (all in `rf_detect\`; git ignores these files):
- `rf_log.csv` - every phone sweep
- `bt_log.csv` - every new headphone seen
- `rf_baseline.json`, `rf_fov.json` - the empty-room baseline and the range test results

Tests (no hardware needed): `py -3.12 -m unittest tests.test_rf_monitor -v`

### Keeping it to this room

Radio has no "distance" setting. Signals just get weaker with distance and walls. So we limit range by **signal strength**: anything quieter than a phone at the back desk counts as "not in this room".

- **Phones: run `fovtest`.** You need one test phone with Wi-Fi off, on a call or streaming video so it keeps transmitting. Put it in three spots:
  1. the **farthest desk** the camera can see (should count)
  2. **behind the camera / antenna** (ignore)
  3. in the **next classroom** or hallway, right behind the wall (ignore; you can skip this one)

  It sets the alarm level halfway between the quietest "count" spot and the loudest "ignore" spot, for each band. It also says honestly if it can't tell them apart.
- **Bluetooth: set `BT_MIN_RSSI`** in `rf_config.py`. Put headphones or the JBL speaker in the next room, run `bluetooth`, read the dBm, and set `BT_MIN_RSSI` about 5 higher. For example, the next room shows -88, so set it to -83.

Tips that help a lot (all free or cheap):
- **Antenna toward the middle of the room**, not in a corner. In a corner, a phone right behind the wall can be closer than your own back desk. A USB extension cable moves the radio without moving the laptop.
- **Aim the antenna's blind spot.** The RTL-SDR's antenna is nearly deaf off the ends of its rods. Lay the rods pointing at the noisiest neighbor's wall.
- **Record the baseline while the classes next door are in session**, so their normal activity counts as "normal".
- **Lower `GAIN`** in `rf_config.py` so far, weak signals disappear. Redo `baseline` and `fovtest` afterwards.

### Camera link (TODO)

The plan: **the radio says *when* a phone is active, and the camera says *which desk*.**

`camera_confirms()` in `rf_detect/rf_monitor.py` is a placeholder; for now it always says "camera not connected". Its comments describe two ways to connect it:
- `app/main.py` saves what the camera sees to a small file, and the RF monitor reads it.
- The RF monitor runs inside `app/main.py` and asks it directly.

Once it's connected, set `CAMERA_CONFIRM = True` in `rf_config.py`, and a radio alert will only count when the camera also sees something.

### RF tuning cheat sheet (`rf_detect/rf_config.py`)

- Too many phone false alarms -> run `fovtest`, raise `THRESHOLD_DB` / `PEAK_THRESHOLD_DB`, or raise `CONFIRM_HITS`
- Missing real phones -> lower `THRESHOLD_DB` / `PEAK_THRESHOLD_DB`, and check the baseline was taken with phones off
- Sweeps too slow -> set bands nobody uses in your area to `"enabled": False`
- B70 / B66 show junk numbers -> set them to `"enabled": False` (they're at the very top of what the radio can reach)
- Headphones from the next room show up -> raise `BT_MIN_RSSI` (for example -85 -> -80)
- Your own headphones keep showing up -> add their name to `BT_IGNORE_NAMES`

### How we decided (radio)

**1. Decode the signals, or just measure them?**
- **Decision: only measure power.** Decoding phone traffic is illegal and creepy. "Is there more signal than normal?" is all we need.

**2. Can the radio show *which desk* the phone is at (a heatmap)?**
- *We tried:* walking the antenna to every desk, and recording a phone at every seat.
- *Problem:* one antenna only hears **how loud**, not **from where**. Two desks the same distance away sound identical. Each phone also changes its own volume up to 1000x depending on its service.
- **Decision: dropped it.** The radio answers "is a phone active right now?" and the camera answers "which desk?". It would need 2+ receivers to work.

**3. Measure the time the signal takes to arrive? Add our own transmitter?**
- *Problem:* you'd need to know when the phone sent the signal, and you can't. Even if you could, the radio's smallest time step (417 ns) is the time light takes to travel 125 m, and the room is 10 m.
- A transmitter would be radar, and transmitting on phone frequencies without a license is illegal.
- **Decision: no.** We use signal strength plus the range test instead.

**4. Bluetooth with the RTL-SDR?**
- *Problem:* Bluetooth is at 2.4 GHz, and the RTL-SDR stops at about 1.77 GHz.
- **Decision: use the laptop's own Bluetooth.** It sees headphones that are switched on, but not who they're connected to. That link is encrypted, and no laptop can watch it.

### Known limits (radio)

- A phone with **bad service transmits up to 1000x louder** than one with good service. So a loud phone in the next room can sometimes get through. The range test reduces this but can't stop it completely.
- **Bands above 1.77 GHz are invisible** to the RTL-SDR: PCS 1.9 GHz, T-Mobile 2.5 GHz and 5G C-band 3.7 GHz. A phone using only those bands is missed.
- A phone in **airplane mode using only Wi-Fi** doesn't use the cell bands at all, so the cell detector can't see it.
- Headphones that are **off, in their closed case, or wired** can't be seen. People can also rename their headphones.
- A crowded room, like a hackathon hall, is very noisy. **Run `baseline` at the venue** right before a demo.

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


