# DT2213 — Polyphonic Finger-Keyboard FM Formant Synth

A real-time gesture-controlled synthesizer built on a webcam + MediaPipe:
your right hand plays 5 notes (one per finger), your left hand picks the
timbre with a gesture, your mouth shapes the vowel, and the horizontal
distance between your hands controls how bright the sound is.

## Requirements

- **Python 3.10 / 3.11 / 3.12** (mediapipe does not yet support 3.13+, and
  3.9 or older is not recommended)
- Webcam + speakers / headphones (microphone is not used, but macOS may
  still ask for permission the first time)
- macOS / Windows / Linux all work

## Run

```bash
python main.py
```

On the first run macOS will ask for camera permission — allow it.
Press **q** in the video window to quit.

## How to play

### Right hand = 5-finger keyboard

| Finger | Note |
|---|---|
| Thumb  | 1 (lowest) |
| Index  | 2 |
| Middle | 3 |
| Ring   | 4 |
| Pinky  | 5 (highest) |

- **Bend a finger** → that note sounds and is held
- **Straighten the finger** → that note releases naturally according to the
  current timbre's ADSR release
- **Bend several fingers at once** → those notes stack into a chord
- **Move the note hand above / level with / below the timbre hand** → play the
  same five notes in high / mid / low registers

### Left hand gesture = pick the timbre

| Gesture | Preset | Character |
|---|---|---|
| ✋ Open palm     | **pad**   | Slow alien choir pad, soft shimmer |
| ✊ Fist          | **key**   | Xenon plucked key, bright inharmonic attack |
| ✌️ Peace sign    | **glass** | Frozen harmonic glass, high resonant bloom |
| ☝️ Index up      | **drone** | Deep atmospheric drone, low sub-octave body |
| 👍 Thumb up      | **rift**  | Retro low-pass saw bass, French-touch filter sweep |

The same gesture must be held for 5 frames before the preset actually
switches (to ignore brief mis-classifications). The preset name tint in
the top-left of the video changes with the preset.

### Mouth = vowel

- **Open jaw**            → raises the first formant F1
- **Smile / pucker**      → moves the second formant F2
- The closest reference vowel (/a/ /e/ /i/ /o/ /u/) is shown on screen

### Horizontal distance between hands = brightness

Hands together → dark, hands apart → bright. A yellow line is drawn
between the two wrists as a visual cue for the distance.

### Vertical distance between hands = register

Note hand higher than timbre hand → high register. Hands level → middle
register. Note hand lower than timbre hand → low register. The register is
read when a finger first bends, so held notes do not slide if your hands move.

## Calibration

While the program is running, watch the terminal output. Three values
matter most:

- **`curl[t i m r p]`** — the real-time curl ratio for each of the 5
  fingers. Fully straight should give values **above 1**, fully bent
  should give values **below ~0.9**.
  If a finger triggers too easily or never triggers, edit the
  `CURL_ON` / `CURL_OFF` arrays near the top of `main.py`.
  Note that the thumb is slightly bent even in a relaxed hand, so its
  threshold (the first element of each list) is intentionally much lower
  than the others.
- **`dist=...`** — the horizontal distance between the wrists. Watch it
  with your hands together and with your hands wide apart, then adjust
  `DIST_MIN, DIST_MAX` accordingly.
- **`dy=...`** — the vertical distance from note-hand wrist to timbre-hand
  wrist. Adjust `REGISTER_HIGH_DY` / `REGISTER_LOW_DY` if high or low register
  is too hard to reach.

## Project layout

```
dt2213_project/
├── main.py                    # All of the code
├── gesture_recognizer.task    # MediaPipe hand-gesture model (~8 MB)
├── face_landmarker.task       # MediaPipe face-mesh model (~3.6 MB)
├── requirements.txt
└── README.md
```
