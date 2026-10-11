# DECISIONS.md - judgment calls made on branch `fix/rf-detection`

The task said: never stop to ask; when something is ambiguous, pick the most
conservative option that keeps current behavior, and write it down here.
Every entry says what I chose, and why.

## Setup

1. **Clone with `core.longpaths=true`.** The first clone failed with "Filename too long"
   (Windows path limit). I retried once with long paths turned on for that clone only (repo-local git config).
2. **Installed into the venv only:** `numpy` (the RF code and tests need it), `yt-dlp` (transcripts),
   and `pyrtlsdr==0.2.93` + `setuptools<81` only so I could read pyrtlsdr's source and use its real
   method names (`set_center_freq`, `read_samples`, `set_agc_mode`). It cannot be imported here
   (no `rtlsdr.dll`), so **no test touches the real driver**.
3. **Transcripts.** Retried each failed download once. Video 2 failed twice with HTTP 429 (rate
   limit); video 3 said "no subtitles for the requested languages" twice. Both are reported as
   "transcript unavailable". I did not use titles, descriptions, other languages or guesses instead.
4. **DECISIONS.md and FIX_REPORT.md live in `rf_detect/`**, next to the code they describe.
5. **Files owned by others** (CLAUDE.md lists camera/vision/rules/ui owners) were not touched.
   Only `rf_detect/`, the RF part of `README.md`, and a new test file changed.

## How "detected" is defined in the new tests

6. **"N dB above the floor"** = the burst's power per Hz is N dB above the noise floor's power per Hz,
   inside the burst's own bandwidth. (Burst + noise together are then slightly more than N dB.)
7. **Burst shapes tested:** 180 kHz (1 LTE resource block), 1.08 MHz (6 RBs), 4.5 MHz (a full 5 MHz
   carrier); 1 ms and 5 ms; landing at random spots in a 10 MHz carrier at 704-714 MHz (B12/17);
   on average 3 bursts per 0.1 s grab. A phone that sends far less often (0.3 per 0.1 s) is
   **measured and reported** in FIX_REPORT.md but **not asserted**, because no power detector that
   only listens to each window 0.1 s per sweep can promise that (see "physical limits").
8. **Synthetic radio realism:** noise stored as 8-bit samples like a real RTL-SDR, a DC spike, and a
   gain that scales everything. The acceptance noise-only test covers **all 8 hearable bands, 50 sweeps,
   with retune junk on**.

## Measuring (rf_monitor.py `measure()`)

9. **Peak statistic = loudest (one 1024-pt FFT slice = 0.43 ms) x (~225 kHz piece) per window.**
   I compared piece widths of 75, 112, 150, 225, 450 and 900 kHz on synthetic bursts.
   The old statistic (6.8 ms x 1.92 MHz average) read a 10 dB, 1 ms wideband burst as only +3.5 dB,
   and a 10 dB 1-RB burst as +0.4 dB. 225 kHz (`PEAK_BINS = 96`) was the best balance: about one
   resource block wide, and its empty-room maximum sits only ~2 dB above the average.
10. **`CHUNKS` kept in rf_config.py but marked "NOT USED"** (no deletions; old notes still make sense).

## Thresholds (the RF math)

11. **`PEAK_THRESHOLD_DB` 6.0 -> 3.0.** This is not a loosening of the same test. The number now
    applies to a *different, finer* measurement:
    - Old peak: average of 16384 samples x ~768 bins. Pure noise varies only ~0.02 dB between sweeps,
      so 6 dB was ~300 standard deviations above noise, but real 1 ms bursts were diluted 7x in time
      (and up to 10x more in frequency) before they got there. A 6 dB burst could never pass.
    - New peak: max of ~2048 pieces (256 slices x 8 pieces), each an average of 96 bins. Measured on the
      synthetic noise: sweep-to-sweep standard deviation of `peak +dB` = **0.17 dB**; the largest value
      in 50 sweeps x 8 bands (400 readings) = **+0.93 dB**.
    - So 3.0 dB is about **12 standard deviations** above the empty-room noise and 2 dB above the worst
      noise reading seen. The extreme-value tail of a max-of-pieces is roughly exponential with scale
      ~0.14 dB, giving roughly exp(-3.0/0.14) ~ 1e-9 chance per window per sweep from noise alone.
    - Meanwhile a 6 dB burst reads +5.1 dB (median, worst shape: 1 RB, 1 ms) up to +7.6 dB; single
      1-RB 1 ms bursts read as low as +4.1 dB in the prototype run (a burst split across two pieces).
      A 4 dB threshold would leave ~0 dB margin for that case, so 3 dB is the highest round number
      that meets the 6 dB requirement with margin.
    - **Real-world cost:** the new measurement is more sensitive to *every* burst, including phones in
      the next room. That is the job of `fovtest`; the test can't prove the room-level behavior.
12. **`THRESHOLD_DB` (average rule) left at 3.0.** The average measurement did not change (noise
    std 0.009 dB), so there was no reason to touch it.
13. **`CONFIRM_HITS = 2`, `CONFIRM_WINDOW = 5` unchanged.** I checked for off-by-one errors and found none
    (the test `test_no_off_by_one` passes on both the old and new code).

## Retune junk

14. **`SETTLE_SAMPLES = 32768` (13.7 ms) thrown away after every retune.** This is a judgment call
    I can't verify without the radio. rtl_power throws away ~2048 samples after a 5 ms sleep; the
    R820T needs a few ms to lock and old samples are still buffered on USB. I picked a generous value.
    Cost: about +1.3 s per sweep (95 windows). To make it checkable, `scan` now measures how long the
    junk really lasts and says if this number is too small.
15. **The simulator's junk (8192 samples = 3.4 ms, 10x louder) is an assumption**, chosen to be longer
    than the old 4096-sample discard. It is labelled as a guess in the code. The tests also run
    without junk, and the main burst fix (entry 9) is proven without it.
16. **The user's description of the junk bug was only partly right.** The old code threw away 4096 samples
    after every retune and an extra 8192 only on the very first read. So the bug was a short and
    uneven discard, not "no discard at all".

## Gain, AGC and saved files

17. **Mismatched baselines are refused, not rescaled.** A baseline/fovtest file now stores the GAIN,
    the sim flag and the measurement settings; any difference = "NOT used, make it again" with a list of
    what changed. Real mode then stops, as it already did with no baseline; sim mode remakes its
    fake baseline, as it already did. Rescaling by the gain difference would be a guess.
18. **Old-format files are refused.** After this update you must run `baseline` and `fovtest` once
    again. That is intended: the old numbers came from the old peak measurement.
19. **`GAIN = "auto"` only gives a warning; it isn't blocked.** That keeps current behavior. With tuner AGC the
    numbers drift, so the warning says to use a number.
20. **The RTL2832 digital AGC is turned off explicitly** with `set_agc_mode(False)` (it exists in
    pyrtlsdr 0.2.93). It is wrapped in try/except so a driver hiccup can't stop the program.

## Tuner range

21. **`TUNER_MAX_MHZ = 1766`.** A window counts as hearable if the top of its usable part is
    <= 1766 MHz. This keeps exactly the same 95 windows as before.
22. **B66 end 1765 -> 1780 MHz** (the real B66 uplink edge). This adds no windows; it only makes the
    tool say "can't hear 1766-1780 MHz" instead of hiding it.
23. **Added a `PCS B2/B25` (1850-1915) row, enabled.** It costs no sweep time and always shows "CAN'T HEAR".
    The other invisible bands (B41 2.5 GHz, n77 3.7 GHz, B30) stay as a comment, to keep the table short.
24. **fovtest skips bands with no hearable windows** (otherwise `max([])` would crash).

## Kept as-is (checked, documented, not changed)

25. **DC notch (60 kHz) kept.** It leaves a 120 kHz hole in the middle of every 1.92 MHz window
    (6% of the spectrum). A burst narrower than ~120 kHz sitting exactly there is missed, and a
    1-RB burst centered there loses about 2/3 of its power. Closing the holes would need overlapping windows,
    which doubles sweep time. That is a trade-off for you to decide, so I only documented it.
26. **Gaps between windows:** 469 Hz uncovered between neighbors (less than one 2.3 kHz bin). Harmless.
27. **B26's last window reaches into B5 (up to 824.7 MHz)**, so a phone at 824 MHz can light up both rows.
    Cosmetic only, not changed.
28. **fovtest threshold math unchanged** ("halfway between front and the loudest ignored spot, at least
    `THRESHOLD_DB`"). It **does** override the normal rules for a band, on purpose. In the
    simulator it set B12's alarm at +9 dB, so in that band a +6 dB phone at the front would be ignored too.
    That is the range gate working as designed, and it is listed as a limit in FIX_REPORT.md.
29. **Simulated monitor:** the fake phone still turns on at sweep 4 with `phone_strength = 4.0` (+12 dB), as before.
30. **Sim files left in place.** Tests leave `rf_baseline_sim.json` / `rf_fov_sim.json` behind
    (gitignored, as before). They are not deleted; the settings check now stops stale ones from being used.
31. **`band_status()` reads `row.get("level")`** so callers that build rows without "level" still work.
    I changed the code, not the test.
32. **Two mistakes in my own new tests were fixed before the first commit.** `repeat(2)` colors the
    noise, so it was replaced with `concatenate`. The window-gap check measured distance between bin
    centers, so it now allows < 2 bin spacings (= less than one bin missing). Neither change loosens
    a requirement on the detector.
