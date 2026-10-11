# FIX_REPORT - RF phone detector (`rf_detect/`), branch `fix/rf-detection`

No RTL-SDR was attached. **Everything below was proven with simulated / synthetic radio
samples only.** Nothing here proves real-room accuracy.

## Confidence

**Proven in simulation (I ran these, and the output is in section 6):**
- Main root cause found and fixed: the old peak measurement averaged every burst over 6.8 ms and the
  whole 1.92 MHz window. A 10 dB, 1 ms burst across the whole window read only **+3.5 dB**, and a 10 dB burst one
  resource block wide read **+0.4 dB**. Both were under the old 6 dB alarm, so a real phone's short bursts could not
  trigger it. With the new measurement, the same 10 dB bursts read about +7 to +10 dB.
- Synthetic bursts **6, 10 and 20 dB** above the floor (1 ms and 5 ms long; 180 kHz, 1.08 MHz and 4.5 MHz
  wide) all reach **ACTIVE within CONFIRM_WINDOW (5) sweeps**. In the characterization run they reached it at sweep 2.
  This also holds with simulated retune junk.
- **Noise only, 50 sweeps x all 8 hearable bands: 0 ACTIVE, 0 "maybe".** The largest `peak +dB` seen was
  +0.93 dB against a 3.0 dB alarm, with a sweep-to-sweep std of 0.17 dB.
- A band above 1766 MHz is reported as **"CAN'T HEAR"** every sweep, not shown as "ok".
- Baselines made at another GAIN, by the simulator, or by the old version are **refused** with the reason shown.
- The settle discard after a retune is now the same after **every** retune.
- `monitor --simulate --sweeps 20 --no-web` exits 0. All bands are "ok" for sweeps 1-3, the fake phone turns
  on at sweep 4 ("maybe", +11.7 dB), and B12/17 is ACTIVE from sweep 5 to 20. No other band ever leaves "ok".

**Likely but unproven (depends on the real radio and the real room):**
- That the burst fix is the reason *your* setup missed phones. It matches the symptom exactly (baseline learned,
  phone not flagged), and the math says real 1 ms LTE bursts were diluted below the threshold.
  But I have never seen your dongle's samples.
- That **retune junk** was hurting you too. In the simulator, junk at the start of each grab inflated the baseline
  peak and hid bursts. The junk length (3.4 ms) is a **guess**. `scan` now measures the real length (checklist step 1).
- That `SETTLE_SAMPLES = 32768` (13.7 ms) is enough. `scan` will say if it isn't.
- That 3.0 dB is a good real-world alarm level. In noise it is very safe (~12 standard deviations), but a real
  room has other phones, other rooms and interference, and those are not Gaussian noise.

**Could still fail in the real room:**
- A phone with good signal transmits very weakly (power control). Its bursts may be under +3 dB at your antenna.
- A phone that is idle, or uses Wi-Fi calling, or only uses PCS / 2.5 GHz / 3.7 GHz, sends nothing the RTL-SDR can hear.
- Each window is listened to for only 0.1 s per sweep (~12-15 s). A phone that sends a burst only every few seconds
  is caught slowly or not at all. In the characterization, sparse phones (0.3 bursts per 0.1 s) were often missed **at any strength**.
- The new measurement is **more sensitive to every phone, including the next room's**. Expect more "maybe"/ACTIVE
  from neighbours until `fovtest` is redone. A `fovtest` level is also a floor: in the simulator it set B12's alarm at
  +9 dB, so a weaker phone at the front of the room is ignored in that band. That is by design, but it is a trade-off.
- B70/B66 sit near the top of the R820T's range. Some dongles give junk there (PLL not locked), and the driver doesn't report it as an error.
- Your first `monitor` after updating will refuse the old `rf_baseline.json` / `rf_fov.json`. That is intended: redo both.

---

## 1. Video takeaways

**s8Q3o74L7VU** - "LTE Cell Scanner and Tracker | LTE eNodeB LTE Cell Tracking using SDR | HackRF | DragonOS Focal"
(auto-subtitles downloaded and read). Only the points that apply to RTL-SDR power-based phone detection:
- The video finds and tracks **downlink** LTE cells (the tower's signal) with the `CellSearch` / `LTE-Tracker` tools,
  which decode the cell ID. This project measures **uplink power only** and never decodes, so the technique itself doesn't apply.
- The presenter limits the search to the frequency range their local carrier actually uses, because scanning a wide
  range is very slow. That applies here: turn off bands nobody uses locally (`"enabled": False`) to shorten each sweep,
  so each window is revisited more often.
- They set the receive gain explicitly (`-g`), and say the tool's default is 40. That matches this tool: use a fixed
  number for `GAIN`, not "auto".
- They say an RTL-SDR user should also set the PPM (frequency correction). For power-only detection a small PPM error
  matters little (tens of kHz against 1.92 MHz windows), but it does shift band edges slightly. `PPM` in rf_config.py exists for this.

**H0vAz96B3uk** - transcript unavailable (HTTP 429 "Too Many Requests", twice).

**XNB8BAdVgNA** - transcript unavailable ("There are no subtitles for the requested languages", twice).

## 2. Bugs found and fixed

Line numbers are from the **original** file (commit 95f7811), with the new location after the arrow.

| file:line | cause | fix | test that proves it |
|---|---|---|---|
| `rf_monitor.py:151-155` -> `measure()` :202 | **Main root cause.** The peak was the mean of 16 chunks of 6.8 ms over the whole 1.92 MHz window. A 1 ms burst was diluted ~7x in time, and a 1-RB burst ~10x in frequency (10 dB 1 ms wideband: +3.5 dB; 10 dB 1 RB: +0.4 dB). | Peak = loudest 0.43 ms FFT slice of ~225 kHz pieces (`PEAK_BINS = 96`). Linear power averaged before dB. | `test_short_burst_shows_in_peak`, `test_6db_and_up_bursts_reach_active` |
| `rf_config.py:33` -> :37 | `PEAK_THRESHOLD_DB = 6.0` was paired with the diluted measurement. A 6 dB burst can never read +6 dB above the baseline peak. | 3.0 dB, justified by the noise statistics (DECISIONS.md #11): noise std 0.17 dB, worst noise reading +0.93 dB. | `test_noise_only_has_no_false_active` (0 ACTIVE / 0 maybe in 50 sweeps x 8 bands) |
| `rf_monitor.py:76-82` -> `read_window()` :89 | After a retune, 12288 samples were discarded only on the very first read, then just 4096 (1.7 ms). Junk left in the grab inflates the baseline peak, so later bursts below it are hidden. | `read_window()`: retune, discard `SETTLE_SAMPLES` (32768) every time, then measure. Real and fake radio both use it. | `test_settles_after_every_retune`, `test_6db_bursts_reach_active_with_retune_junk` |
| `rf_monitor.py:215-225` -> `measure_settings()` :274, `load_baseline()` :351 | Baseline/fovtest files didn't record GAIN or anything else. A baseline from another gain (README tells you to lower GAIN), from the simulator, or from the old measurement loaded silently. At 10 dB less gain, every delta is -10 dB and nothing alarms. | Files store gain, sim flag, measure version and FFT/settle/notch settings. A mismatch is refused, with the differences printed. Also warns about windows with no baseline. | `test_baseline_from_other_gain_is_refused`, `test_sim_baseline_is_never_used_for_the_real_radio`, `test_old_files_are_not_trusted` |
| `rf_monitor.py:73` -> `RealRadio.__init__` | `GAIN = "auto"` (tuner AGC) was accepted silently, and the RTL2832 digital AGC was never set. Levels can drift between baseline and monitor. | Warn on "auto"; `set_agc_mode(False)` explicitly. | not testable without hardware (checklist step 2) |
| `rf_monitor.py:122-123` -> `make_steps()` :174, `cant_hear()` :182 | Windows above the 1766 MHz tuner limit were swept anyway (junk on hardware, "ok" in the sim). | Skipped, and reported as "CAN'T HEAR" (whole band) or "can't hear X-Y MHz" (part of a band). | `test_band_above_tuner_limit_is_reported_unhearable` |
| `rf_config.py:101` | B66 was cut at 1765 MHz, but its uplink runs to 1780, which hid the 1766-1780 MHz blind spot. PCS (the biggest US uplink band) wasn't listed. | B66 now ends at 1780; a PCS B2/B25 row was added. Both show what can't be heard. Hearable windows unchanged (95). | `test_monitor_simulate_20_sweeps` (checks the CAN'T HEAR row) |
| `rf_monitor.py:98-105` -> `FakeRadio` :99 | The simulator was unrealistic: an 8 ms, +12 dB burst across the **whole** window in **every** grab, no retune junk, no tuner limit. It passed with the bug, which hid the bug. | 1-5 ms bursts of 1-6 RBs on a 10 MHz carrier, ~4 per 0.1 s, retune junk, tuner limit, same `read_window()` as the real radio. | `test_monitor_simulate_20_sweeps` (the old code fails it) |
| `rf_monitor.py:540` | The "no baseline" message always named the real baseline file. | Names the file actually used. | (message only) |
| fovtest (new) | Would crash on a band with no hearable windows (`max([])`). | Such bands are skipped. | `test_fovtest_then_monitor` (with the new PCS row) |

**Checked, no bug found (guard tests added):**
- dB vs linear: averages are taken in linear power before dB, and medians of dB are fine (`test_average_is_linear_power`).
- DC notch and USABLE_FRACTION bin indexing: 384 bins kept on each side; 469 Hz uncovered between windows (< 1 bin) (`test_bins_and_windows_line_up`).
- Confirm window: no off-by-one (`test_no_off_by_one`).
- dB math in `band_readings` / `step_heat` / fovtest medians: correct.

**Checked, documented, not changed** (see DECISIONS.md): the 120 kHz DC-notch hole in the middle of every window; B26's last
window overlapping B5; the fovtest level replacing the normal rules for its band (range gate by design).

## 3. DECISIONS.md entries

All 32 entries are in `rf_detect/DECISIONS.md`. In short:
1. Cloned with `core.longpaths=true` (Windows path limit).
2. venv only: numpy, yt-dlp, pyrtlsdr 0.2.93 (for reading its source; it can't load without the DLL).
3. Two transcripts unavailable after one retry each; no guessing.
4. Docs live in `rf_detect/`.
5. Files owned by teammates were not touched.
6. "N dB above floor" = burst power per Hz N dB above noise power per Hz, in the burst's own bandwidth.
7. Burst shapes tested: 180 kHz / 1.08 MHz / 4.5 MHz, 1 and 5 ms, ~3 per 0.1 s. Sparse phones are measured, not asserted.
8. Synthetic samples use 8-bit numbers, a DC spike and a gain; the noise test covers all bands, 50 sweeps, with junk.
9. Peak = 0.43 ms x 225 kHz pieces, chosen by comparing 6 piece widths.
10. `CHUNKS` kept but marked unused.
11. `PEAK_THRESHOLD_DB` 6 -> 3, with the noise math (std 0.17 dB, max +0.93 dB, ~1e-9 per window).
12. `THRESHOLD_DB` unchanged.
13. Confirm settings unchanged (no off-by-one).
14. `SETTLE_SAMPLES` = 32768 is a generous guess (+1.3 s per sweep), checked by `scan`.
15. The simulator's 3.4 ms junk is an assumption; the main fix is proven without it.
16. The old code did discard 4096 per retune; the bug was a short and uneven discard.
17. Mismatched baselines are refused, not rescaled.
18. Old-format files are refused: redo baseline and fovtest once.
19. `GAIN = "auto"`: warning only.
20. RTL2832 AGC turned off explicitly (wrapped in try).
21. `TUNER_MAX_MHZ = 1766`; a window counts if its usable top is <= 1766 (same 95 windows).
22. B66 end set to 1780.
23. PCS row added; other invisible bands stay as a comment.
24. fovtest skips unhearable bands.
25. DC notch kept (closing the holes doubles sweep time).
26. 469 Hz window gaps are harmless.
27. B26/B5 overlap is cosmetic.
28. fovtest math unchanged; it can raise a band's alarm above the normal rules.
29. Sim phone unchanged (+12 dB from sweep 4).
30. Sim files not deleted; the settings check protects against stale ones.
31. `band_status` tolerates rows without "level" (I changed the code, not the test).
32. Two mistakes in my own new tests were fixed before the first commit, without loosening anything.

## 4. Remaining physical limits

- **Phone power control.** A phone near its tower may transmit around -40 dBm instead of its +23 dBm maximum
  (a ~60 dB range; the README's "1000x" = 30 dB is conservative). A well-served phone at the back desk can sit near
  the noise floor, while a badly-served phone in the next room can be louder. One antenna can't tell distance from loudness.
- **Idle phones barely transmit.** A phone in a pocket with the screen off sends almost nothing on the uplink (occasional
  short reports, location updates minutes apart). This detector catches phones **being used**: streaming, calls,
  messaging, uploads. In the characterization, a phone sending ~0.3 bursts per 0.1 s was often missed even at +20 dB.
- **Listening time.** Each 1.92 MHz window is heard for 0.11 s out of every ~12-15 s sweep (95 windows), which is under 1%
  of the time. Turning off unused bands shortens the sweep and raises this.
- **Tuner range.** The R820T stops around 1766 MHz. It can't hear PCS 1850-1915 (B2/B25), B66 above 1766, B41/n41 2.5 GHz,
  n77 3.7 GHz, B30, CBRS, Wi-Fi or Bluetooth. A phone using only those is missed. The table now says "can't hear".
- **Near the top of the range**, B70/B66 sensitivity is lower and some dongles lose PLL lock without reporting an error.
- **DC notch holes:** 120 kHz in the middle of every window (6% of the spectrum).
- **Wi-Fi calling / airplane mode + Wi-Fi:** no cell uplink at all.
- **8-bit ADC:** a phone very close to the antenna at high GAIN can clip. It still shows as more power, but the dB number is too low.

## 5. Hardware checklist (run with the real RTL-SDR, Windows cmd, from the repo root)

Close SDR++ first. Keep `GAIN = 40` (or whatever you pick) the same for every step. If you change it, redo steps 2-5.
**I could not check these expected readings on hardware.** They come from the math and the simulator. Write down what you actually see.

1. **Radio + retune check (phones in the room don't matter yet):**
   `py -3.12 rf_detect\rf_monitor.py scan --no-web`
   - Expect: `Watching 9 bands in 95 frequency windows`, plus two "can't hear" lines (B66 1766-1780, PCS).
   - Expect: a table with a `level dB` for every band except PCS. Values are in dBFS (radio full scale), roughly similar
     across bands, maybe a few dB lower at B70/B66. The `+dB` columns are empty ("no baseline").
   - Expect at the end: `Retune check: junk after tuning lasted about X ms. We throw away 13.7 ms.` then `OK`.
     If it says `WARNING: not enough!`, set `SETTLE_SAMPLES` to the number it gives, then run `scan` again.
     If it says `never settled`, check the antenna/USB and rerun. A strong signal turning on and off can also cause it.
   - If the console prints `[R82XX] PLL not locked!`, set B70/B66 to `"enabled": False`.
2. **Empty-room baseline (every phone off or in airplane mode, room empty):**
   `py -3.12 rf_detect\rf_monitor.py baseline --no-web`
   - Expect: `Saved ...rf_baseline.json with 95 frequency windows`.
   - Expect: `Empty-room check`, with each band about **+2 dB** (the simulator gives +1.9 to +2.0, worst window +2.2).
     If a band reads **more than about +4 dB**, something is transmitting in the "empty" room, or junk is getting through.
     Find it before going on, or take the baseline at a quieter time.
3. **Empty-room sanity monitor (phones still off):**
   `py -3.12 rf_detect\rf_monitor.py monitor --sweeps 10 --no-web`
   - Expect: every hearable band `ok`, `avg +dB` within about +/-0.5, `peak +dB` mostly under +1
     (the simulator's worst was +0.93). Any `maybe` here means the room isn't quiet, which is real-world noise
     the simulator can't model. Note which band. It may be a neighbor's phone.
4. **Positive control (one test phone, Wi-Fi OFF, on a call or video call, 1-2 m from the antenna):**
   `py -3.12 rf_detect\rf_monitor.py monitor --sweeps 10 --no-web`
   - Expect: within 2-3 sweeps, one band goes `maybe` then `ACTIVE`, with `peak +dB` well above +3.
     I can't predict the exact dB: it depends on the phone's power control (could be +5, could be +30).
   - If nothing happens: check which band the phone is using (a network-info / field-test app shows the LTE band;
     iPhone field test `*3001#12345#*`). If it's PCS (B2/B25), B41 or n77, the RTL-SDR can't hear it, and the table
     should say "can't hear". Try another phone or carrier.
   - Then the same phone **idle, screen off**: expect mostly `ok`. That's the idle-phone limit, not a bug.
5. **Range test:** `py -3.12 rf_detect\rf_monitor.py fovtest` (follow the prompts: farthest desk, behind camera, next room)
   - Expect: for the phone's band, `OK: alarm at +X dB` with a gap of at least 3 dB. Other bands: `phone not seen`.
   - **Write down X.** In that band, any phone weaker than +X at the antenna is ignored, even at the front desk.
     If X is much higher than what a texting (not calling) phone shows in step 4, the range gate will hide it.
6. **Real monitor:** `py -3.12 rf_detect\rf_monitor.py monitor`
   - Expect: `Only counting phones in front of the camera for: <band>`. Test phone at the farthest desk on a call should give `ACTIVE`;
     behind the camera, `ok` or `maybe`. Check `rf_detect\rf_log.csv` afterwards.
7. **After changing GAIN, SAMPLE_RATE, SETTLE_SAMPLES, PEAK_BINS or the antenna:** `monitor` will refuse the old files
   and say why. Redo steps 2 and 5.

## 6. Final test output and git log

Full suite (`venv\Scripts\python -m unittest tests.test_rf_monitor tests.test_rf_detection -v`), last run:

```
test_finds_fake_headphones_but_not_the_mouse ... ok
test_headphone_rules ... ok
test_fovtest_then_monitor ... ok
test_phone_behind_the_camera_is_ignored ... ok
test_runs_as_a_module_from_the_repo_root ... ok
test_runs_by_path_from_the_repo_root ... ok
test_runs_from_inside_the_rf_detect_folder ... ok
test_scan_writes_the_waterfall ... ok
test_survey_mode_is_gone ... ok
test_6db_and_up_bursts_reach_active ... ok
test_6db_bursts_reach_active_with_retune_junk ... ok
test_band_above_tuner_limit_is_reported_unhearable ... ok
test_noise_only_has_no_false_active ... ok
    noise only, 50 sweeps: 0 ACTIVE, 0 single 'maybe' hits
test_no_off_by_one ... ok
test_average_is_linear_power ... ok
test_bins_and_windows_line_up ... ok
test_short_burst_shows_in_peak ... ok
test_settles_after_every_retune ... ok
test_baseline_from_other_gain_is_refused ... ok
test_old_files_are_not_trusted ... ok
test_sim_baseline_is_never_used_for_the_real_radio ... ok
test_monitor_simulate_20_sweeps ... ok
----------------------------------------------------------------------
Ran 22 tests in 311.746s

OK
```
(Class prefixes are trimmed. Baseline progress lines printed during the tests are left out.) On the **original** code, 9 of the
13 new tests failed for the reasons in section 2. Two other failures were mistakes in my tests, which I fixed (DECISIONS.md #32).
The linear-average, bin, confirm-window and noise-only tests passed on the original code too: those parts had no bug.

`py -3.12 rf_detect\rf_monitor.py monitor --simulate --sweeps 20 --no-web` (exit 0), B12/17 row per sweep:

```
Watching 9 bands in 95 frequency windows
  B66 AWS (1710-1780 MHz): can't hear 1766-1780 MHz. A phone using only this part is MISSED.
  PCS B2/B25 (1850-1915 MHz): CAN'T HEAR: above the 1766 MHz RTL-SDR limit. A phone using only this part is MISSED.
(SIMULATION: fake radio, no hardware)
Only counting phones in front of the camera for: B12/17 700      <- rf_fov_sim.json from the test run (alarm +9.0 dB)
sweep 1   B12/17 700  -37.0  +0.0   +0.3  ok
sweep 3   B12/17 700  -37.0  +0.0   +0.2  ok
sweep 4   B12/17 700  -35.7  +1.3  +11.7  maybe        <- fake phone turns on
sweep 5   B12/17 700  -36.3  +0.6  +12.4  ACTIVE  <-- transmitting
...       (ACTIVE in every sweep 5-20, 16 rows; no other band ever leaves "ok")
sweep 20  B12/17 700  -36.0  +1.0  +11.4  ACTIVE  <-- transmitting
PCS B2/B25 (every sweep)               CAN'T HEAR: above the 1766 MHz RTL-SDR limit
Bands that look ACTIVE right now: ['B12/17 700']
```
Note: the 0.1 s **average** only rose about +0.5 to +1.6 dB, below the 3 dB average rule. Only the new burst peak catches this phone.

Characterization (synthetic, B12/17, retune junk on, 10 sweeps each; full table in the appendix below):

| burst | 3 per 0.1 s: suspicious sweeps | first ACTIVE | 0.3 per 0.1 s: suspicious sweeps |
|---|---|---|---|
| 3 dB, 180 kHz, 1 ms | 3/10 | sweep 4 | 0/10 |
| 3 dB, 1.08 MHz, 1 ms | 10/10 | 2 | 4/10 |
| 6 dB, 180 kHz, 1 ms | 10/10 | 2 | 1/10 |
| 6 dB, 4.5 MHz, 1 ms | 10/10 | 2 | 6/10 |
| 10 dB, 1.08 MHz, 5 ms | 10/10 | 2 | 5/10 |
| 20 dB, 180 kHz, 1 ms | 10/10 | 2 | 3/10 |

`git log --oneline` for `fix/rf-detection` (newest first; the commit that adds this report sits on top):

```
157b47e Docs: README for the new checks and limits, DECISIONS.md
ddd1d36 Fix the simulator hiding the bug; add retune and empty-room self-checks
ffcebc6 Fix: report bands above the 1766 MHz tuner limit instead of hiding them
e99613a Fix: never use a baseline/fovtest made with other settings or the simulator
534f615 Fix: settle the radio after every retune, not only the first
36a1acc Fix: short phone bursts were averaged away in the peak measurement
5221511 Add synthetic-IQ detection tests (failing on current code)
95f7811 Merge pull request #8 from nerdyvinny/remove-desk-boxes-vrnc4s   <- starting point
```
Nothing was pushed. No PR, no remote touched. Fix-and-test iterations used: 6 of 8.

## Appendix: full characterization run

Synthetic SDR, B12/17 only, retune junk 8192 samples, baseline of 5 sweeps, then 10 sweeps.
`suspicious_sweeps/10` = sweeps where `is_suspicious` was true; `first_ACTIVE` = first sweep with 2 hits in the last 5.
Rows with `bursts/0.1s = 0.3` are the sparse (barely used) phone. They are reported, not required to pass.

```
NOISE ONLY 50 sweeps x 8 bands: peak_delta max +0.93 dB, std 0.17; avg_delta max +0.05 dB, std 0.009
snr_dB width   dur  bursts/0.1s  suspicious_sweeps/10  first_ACTIVE  median_peak_delta
    3    180k  1ms    3.0         3/10              4          +2.6
    3    180k  5ms    3.0         5/10              5          +3.2
    3   1080k  1ms    3.0        10/10              2          +3.9
    3   1080k  5ms    3.0        10/10              2          +4.2
    3   4500k  1ms    3.0        10/10              2          +4.3
    3   4500k  5ms    3.0        10/10              2          +4.8
    4    180k  1ms    3.0         5/10              5          +3.1
    4    180k  5ms    3.0         7/10              2          +3.5
    4   1080k  1ms    3.0        10/10              2          +4.4
    4   1080k  5ms    3.0        10/10              2          +5.0
    4   4500k  1ms    3.0        10/10              2          +4.9
    4   4500k  5ms    3.0        10/10              2          +6.8
    5    180k  1ms    3.0         9/10              3          +3.5
    5    180k  5ms    3.0         9/10              2          +4.7
    5   1080k  1ms    3.0        10/10              2          +5.2
    5   1080k  5ms    3.0        10/10              2          +5.5
    5   4500k  1ms    3.0        10/10              2          +5.5
    5   4500k  5ms    3.0        10/10              2          +6.7
    6    180k  1ms    3.0        10/10              2          +5.1
    6    180k  5ms    3.0        10/10              2          +5.4
    6   1080k  1ms    3.0        10/10              2          +5.7
    6   1080k  5ms    3.0         9/10              2          +6.4
    6   4500k  1ms    3.0        10/10              2          +6.4
    6   4500k  5ms    3.0        10/10              2          +7.6
    8    180k  1ms    3.0        10/10              2          +6.0
    8    180k  5ms    3.0        10/10              2          +6.7
    8   1080k  1ms    3.0        10/10              2          +7.6
    8   1080k  5ms    3.0        10/10              2          +8.1
    8   4500k  1ms    3.0        10/10              2          +8.3
    8   4500k  5ms    3.0        10/10              2          +9.7
   10    180k  1ms    3.0        10/10              2          +7.3
   10    180k  5ms    3.0        10/10              2          +8.6
   10   1080k  1ms    3.0        10/10              2          +9.5
   10   1080k  5ms    3.0        10/10              2          +10.0
   10   4500k  1ms    3.0        10/10              2          +9.7
   10   4500k  5ms    3.0        10/10              2          +11.8
   20    180k  1ms    3.0        10/10              2          +17.6
   20    180k  5ms    3.0        10/10              2          +18.0
   20   1080k  1ms    3.0        10/10              2          +19.0
   20   1080k  5ms    3.0        10/10              2          +19.8
   20   4500k  1ms    3.0        10/10              2          +19.8
   20   4500k  5ms    3.0        10/10              2          +21.8
    3    180k  1ms    0.3         0/10              never          +0.4
    3    180k  5ms    0.3         0/10              never          +0.4
    3   1080k  1ms    0.3         4/10              3          +1.7
    3   1080k  5ms    0.3         4/10              4          +0.5
    3   4500k  1ms    0.3         8/10              2          +3.9
    3   4500k  5ms    0.3         6/10              3          +4.1
    4    180k  1ms    0.3         0/10              never          +0.5
    4    180k  5ms    0.3         2/10              3          +0.4
    4   1080k  1ms    0.3         4/10              4          +0.6
    4   1080k  5ms    0.3         5/10              3          +2.3
    4   4500k  1ms    0.3         7/10              3          +4.6
    4   4500k  5ms    0.3         5/10              4          +2.8
    5    180k  1ms    0.3         2/10              never          +0.4
    5    180k  5ms    0.3         3/10              3          +0.4
    5   1080k  1ms    0.3         2/10              8          +0.3
    5   1080k  5ms    0.3         2/10              10          +0.4
    5   4500k  1ms    0.3         8/10              3          +4.7
    5   4500k  5ms    0.3         8/10              2          +5.4
    6    180k  1ms    0.3         1/10              never          +0.5
    6    180k  5ms    0.3         6/10              3          +3.6
    6   1080k  1ms    0.3         4/10              4          +0.5
    6   1080k  5ms    0.3         4/10              2          +0.6
    6   4500k  1ms    0.3         6/10              2          +5.6
    6   4500k  5ms    0.3         5/10              3          +3.0
    8    180k  1ms    0.3         3/10              6          +0.6
    8    180k  5ms    0.3         1/10              never          +0.3
    8   1080k  1ms    0.3         6/10              5          +5.6
    8   1080k  5ms    0.3         2/10              never          +0.4
    8   4500k  1ms    0.3         9/10              2          +7.7
    8   4500k  5ms    0.3         7/10              2          +8.0
   10    180k  1ms    0.3         3/10              10          +0.4
   10    180k  5ms    0.3         4/10              5          +0.5
   10   1080k  1ms    0.3         2/10              never          +0.3
   10   1080k  5ms    0.3         5/10              2          +2.2
   10   4500k  1ms    0.3         3/10              8          +0.4
   10   4500k  5ms    0.3         6/10              3          +9.2
   20    180k  1ms    0.3         3/10              10          +0.4
   20    180k  5ms    0.3         2/10              9          +0.3
   20   1080k  1ms    0.3         5/10              3          +9.4
   20   1080k  5ms    0.3         3/10              3          +0.5
   20   4500k  1ms    0.3         7/10              2          +18.7
   20   4500k  5ms    0.3         8/10              2          +19.3
```
