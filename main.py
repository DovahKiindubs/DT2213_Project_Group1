"""
DT2213 project: left-hand 5-finger keyboard + right-hand gesture preset switch
on an FM Formant synthesizer.

Architecture:
  Each finger = one Voice (FM osc A carrier + osc B modulator + its own ADSR).
  All voices are summed and run through two formant band-pass resonators
  (mouth shape -> F1/F2), so the whole thing sings vowels like a choir.

Controls:
  Left hand, 5 fingers       -> 5 pentatonic notes (thumb lowest, pinky highest)
  Curl / extend a finger     -> that note's note-on / note-off (held while bent)
  Curl multiple fingers      -> chord (multiple voices sound at once)
  Right hand gesture         -> switch timbre preset:
      Open_Palm  (Open Palm) -> pad    (slow alien choir pad)
      Closed_Fist (Fist)     -> key    (xenon plucked key)
      Victory     (Victory)  -> glass  (frozen harmonic glass)
      Pointing_Up (Point)    -> drone  (deep atmospheric drone)
  Horizontal distance between the two hands -> timbre brightness (MOD_INDEX)
  Jaw open      (Face)       -> formant F1
  Smile / pucker (Face)      -> formant F2

The camera frame is mirrored so left/right hand labels match the user's view.
Press q to quit.
"""

import os
import time

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
from scipy.signal import iirpeak, lfilter

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Model files live next to this script, loaded by absolute path
HERE = os.path.dirname(os.path.abspath(__file__))
GESTURE_MODEL = os.path.join(HERE, "gesture_recognizer.task")
FACE_MODEL = os.path.join(HERE, "face_landmarker.task")

# ===== Audio parameters =====
SAMPLE_RATE = 44100
# Smaller block -> shorter trigger-to-sound latency
# Too small risks underruns (audible clicks). 256 = ~5.8 ms, 128 = ~2.9 ms.
BLOCK_SIZE = 256
MASTER_AMP = 0.2  # master volume. No soft-clipper: keep this small enough
# that a 5-voice bright chord stays mostly under ±1, so the
# output is fully linear (no waveshaping distortion).

# ===== FM global parameters =====
# Modulation index range; the actual value is controlled by hand distance
# (see DIST_MIN/MAX below) and feeds every voice's FM depth.
MOD_INDEX_MIN, MOD_INDEX_MAX = 0.0, 15.0

# ===== Timbre presets (4 alien / atmospheric flavours) =====
# Switching a preset rewrites every voice's ADSR parameters and the shared
# voice colour in one shot. attack/decay/release are seconds, sustain is 0..1.
# mod_ratio = modulator freq / carrier freq.
# fm_scale multiplies hand-distance brightness per preset.
# second_harmonic / sub_mix add fixed layers so the four presets do not all
# feel like the same sine-FM patch with different envelopes.
PRESETS = {
    "pad": {
        "amp_attack": 1.80,
        "amp_decay": 2.50,
        "amp_sustain": 0.82,
        "amp_release": 5.50,
        "mod_attack": 2.80,
        "mod_decay": 2.00,
        "mod_sustain": 0.38,
        "mod_release": 4.50,
        "mod_ratio": 0.50,
        "fm_scale": 0.42,
        "second_harmonic": 0.18,
        "sub_mix": 0.08,
        "tremolo_rate": 0.09,
        "tremolo_depth": 0.12,
        "dry_mix": 0.10,
        "f1_mix": 0.55,
        "f2_mix": 0.45,
        "filter_q": 5.0,
        "gain": 0.82,
    },
    "key": {
        # Ethereal, fully harmonic key. Soft pluck onset that blooms into a
        # long shimmering tail; no inharmonic clang.
        "amp_attack": 0.012,
        "amp_decay": 1.20,
        "amp_sustain": 0.22,
        "amp_release": 3.50,
        # Brightness blooms in quickly then settles to a clean, soft body.
        "mod_attack": 0.008,
        "mod_decay": 1.10,
        "mod_sustain": 0.06,
        "mod_release": 2.20,
        # Integer mod_ratio = pure harmonic spectrum (octave overtones).
        # This is the key change vs the old metallic 2.414 ratio.
        "mod_ratio": 2.0,
        "fm_scale": 0.50,
        "second_harmonic": 0.18,
        "sub_mix": 0.20,
        # Subtle slow shimmer for the ethereal feel.
        "tremolo_rate": 0.18,
        "tremolo_depth": 0.07,
        "dry_mix": 0.10,
        "f1_mix": 0.48,
        "f2_mix": 0.52,
        "filter_q": 5.5,
        "gain": 0.68,
    },
    "glass": {
        "amp_attack": 0.12,
        "amp_decay": 1.80,
        "amp_sustain": 0.42,
        "amp_release": 4.00,
        "mod_attack": 0.03,
        "mod_decay": 3.00,
        "mod_sustain": 0.20,
        "mod_release": 3.20,
        "mod_ratio": 3.01,
        "fm_scale": 0.80,
        "second_harmonic": 0.12,
        "sub_mix": 0.00,
        "tremolo_rate": 0.23,
        "tremolo_depth": 0.08,
        "dry_mix": 0.05,
        "f1_mix": 0.28,
        "f2_mix": 0.72,
        "filter_q": 8.0,
        "gain": 0.65,
    },
    "drone": {
        "amp_attack": 4.50,
        "amp_decay": 3.00,
        "amp_sustain": 0.95,
        "amp_release": 8.00,
        "mod_attack": 6.00,
        "mod_decay": 2.00,
        "mod_sustain": 0.72,
        "mod_release": 8.00,
        "mod_ratio": 1.333,
        "fm_scale": 0.55,
        "second_harmonic": 0.08,
        "sub_mix": 0.45,
        "tremolo_rate": 0.04,
        "tremolo_depth": 0.18,
        "dry_mix": 0.12,
        "f1_mix": 0.70,
        "f2_mix": 0.30,
        "filter_q": 3.2,
        "gain": 0.70,
    },
}
DEFAULT_PRESET = "pad"

# Right-hand gesture -> preset name. Anything else leaves the preset unchanged.
GESTURE_TO_PRESET = {
    "Open_Palm": "pad",  # open palm  -> slow alien choir pad
    "Closed_Fist": "key",  # fist       -> xenon plucked key
    "Victory": "glass",  # peace sign -> frozen harmonic glass
    "Pointing_Up": "drone",  # index up   -> deep atmospheric drone
}
# Same gesture must be stable for N frames before the preset actually switches,
# to avoid flicker on momentary mis-classification.
PRESET_DEBOUNCE_FRAMES = 5

# HUD tint per preset (BGR for cv2.putText).
PRESET_COLORS = {
    "pad": (180, 255, 200),  # mint
    "key": (255, 220, 180),  # icy blue
    "glass": (220, 255, 220),  # pale aqua
    "drone": (255, 180, 220),  # lavender
}

# ===== Finger keyboard =====
PENTATONIC_STEPS = [0, 2, 4, 7, 9]  # major pentatonic semitone offsets
SCALE_ROOT_HZ = 65.41  # C2 -- lowest possible note
KEYBOARD_OCTAVE = 1  # which octave (0-based) the 5 notes sit in

# Per-finger curl thresholds with hysteresis.
# Curl ratio < CURL_ON  -> finger is now bent (note-on).
# Curl ratio > CURL_OFF -> finger is now straight (note-off).
# Order: thumb / index / middle / ring / pinky.
# The thumb is slightly bent even in a relaxed hand, so its threshold must
# be much lower -- only a deliberate, deep bend should trigger it.
CURL_ON = [0.70, 0.95, 0.95, 0.95, 0.95]
CURL_OFF = [0.75, 1.00, 1.00, 1.00, 1.00]

# Formant centre-frequency ranges (Hz), swept by mouth shape.
F1_MIN, F1_MAX = 200.0, 1600.0
F2_MIN, F2_MAX = 500.0, 3200.0

# Default resonator quality factor. Presets can override this; higher Q =
# narrower, more "vowel-like".
FILTER_Q = 4.0

# Horizontal distance between the two wrists (normalised image coords) mapped
# to MOD_INDEX. Hands together -> dark, hands apart -> bright.
# Watch the printed `dist=` value to calibrate.
DIST_MIN, DIST_MAX = 0.15, 0.75

# Reference vowel formants, used only to label the nearest vowel on screen.
VOWEL_REF = {
    "i": (240, 2400),
    "e": (390, 2300),
    "a": (860, 1600),
    "o": (360, 630),
    "u": (240, 560),
}


def note_freq(degree, octave):
    """Pentatonic degree (1..5) in the given octave (0-based) -> frequency (Hz)."""
    step = PENTATONIC_STEPS[degree - 1]
    return SCALE_ROOT_HZ * 2.0 ** (octave + step / 12.0)


# Five fixed note frequencies, one per finger (thumb=lowest, pinky=highest).
NOTE_FREQS = [note_freq(d, KEYBOARD_OCTAVE) for d in range(1, 6)]

# MediaPipe hand-landmark indices of the 5 fingertips (thumb..pinky).
FINGER_TIPS = [4, 8, 12, 16, 20]


class ADSR:
    """Linear ADSR envelope generator. Each call to process() advances one block.

    attack / decay / sustain / release can be mutated at any time
    (apply_preset() rewrites them when the user switches timbre).
    """

    def __init__(self, sample_rate, attack, decay, sustain, release):
        self.sr = sample_rate
        self.attack = attack
        self.decay = decay
        self.sustain = sustain
        self.release = release
        self.state = "idle"
        self.value = 0.0

    def process(self, frames, gate):
        out = np.empty(frames, dtype=np.float64)
        # Per-sample increments (max(1,...) guards against zero-second segments).
        a_inc = 1.0 / max(1.0, self.attack * self.sr)
        d_inc = (1.0 - self.sustain) / max(1.0, self.decay * self.sr)
        r_inc = 1.0 / max(1.0, self.release * self.sr)

        for i in range(frames):
            if gate:
                if self.state in ("idle", "release"):
                    self.state = "attack"
                if self.state == "attack":
                    self.value += a_inc
                    if self.value >= 1.0:
                        self.value = 1.0
                        self.state = "decay"
                elif self.state == "decay":
                    self.value -= d_inc
                    if self.value <= self.sustain:
                        self.value = self.sustain
                        self.state = "sustain"
                else:  # sustain
                    self.value = self.sustain
            else:
                if self.state != "idle":
                    self.state = "release"
                if self.state == "release":
                    self.value -= r_inc
                    if self.value <= 0.0:
                        self.value = 0.0
                        self.state = "idle"
            out[i] = self.value
        return out


class Voice:
    """One sounding note: FM (A carrier + B modulator) + its own ADSR + phases.

    note_on / note_off are called from the vision thread; render() is called
    once per block from the audio thread.
    """

    def __init__(self, sample_rate):
        self.sr = sample_rate
        self.freq = 220.0
        self.gate = False
        self._phase_c = 0.0
        self._phase_m = 0.0
        self._phase_sub = 0.0
        self._phase_lfo = 0.0
        # ADSR params get overwritten immediately by apply_preset(); these
        # initial values just have to be non-degenerate.
        self.amp_env = ADSR(sample_rate, 0.5, 1.5, 0.7, 2.5)
        self.mod_env = ADSR(sample_rate, 1.0, 0.6, 0.8, 2.0)

    def note_on(self, freq):
        self.freq = freq
        self.gate = True

    def note_off(self):
        self.gate = False

    def render(self, frames, mod_index, preset):
        """Render this voice's FM signal * amp_env for one block (length=frames)."""
        amp_env = self.amp_env.process(frames, self.gate)
        mod_env = self.mod_env.process(frames, self.gate)
        mod_ratio = preset["mod_ratio"]
        fm_depth = mod_index * preset["fm_scale"]
        # Frequency is constant within a block, so plain arithmetic is enough
        # (no need for cumsum, which we'd only need for varying freq).
        n = np.arange(1, frames + 1)
        dphi_c = 2.0 * np.pi * self.freq / self.sr
        dphi_m = 2.0 * np.pi * self.freq * mod_ratio / self.sr
        dphi_sub = 0.5 * dphi_c
        phase_c = self._phase_c + dphi_c * n
        phase_m = self._phase_m + dphi_m * n
        phase_sub = self._phase_sub + dphi_sub * n
        self._phase_c = phase_c[-1] % (2.0 * np.pi)
        self._phase_m = phase_m[-1] % (2.0 * np.pi)
        self._phase_sub = phase_sub[-1] % (2.0 * np.pi)

        fm = fm_depth * mod_env * np.sin(phase_m)
        source = np.sin(phase_c + fm)

        second_harmonic = preset["second_harmonic"]
        if second_harmonic:
            source += second_harmonic * np.sin(2.0 * phase_c + 0.55 * fm)

        sub_mix = preset["sub_mix"]
        if sub_mix:
            source += sub_mix * np.sin(phase_sub)

        normaliser = 1.0 + abs(second_harmonic) + abs(sub_mix)
        source /= normaliser

        tremolo_depth = preset["tremolo_depth"]
        tremolo_rate = preset["tremolo_rate"]
        if tremolo_depth and tremolo_rate:
            dphi_lfo = 2.0 * np.pi * tremolo_rate / self.sr
            phase_lfo = self._phase_lfo + dphi_lfo * n
            self._phase_lfo = phase_lfo[-1] % (2.0 * np.pi)
            lfo = 0.5 + 0.5 * np.sin(phase_lfo)
            source *= (1.0 - tremolo_depth) + tremolo_depth * lfo

        return source * amp_env


class FormantSynth:
    """N voices summed -> two formant band-pass filters -> output.

    Several voices gated at once = a chord; each finger drives exactly one voice.
    F1/F2 (vowel), mod_index (brightness) and the active timbre preset are
    shared across all voices.
    """

    def __init__(self, sample_rate, num_voices=5):
        self.sr = sample_rate
        self.voices = [Voice(sample_rate) for _ in range(num_voices)]

        # Shared formants + modulation index. Vision thread writes target_*;
        # the callback smooths the current value towards target every block.
        self.f1 = 600.0
        self.target_f1 = 600.0
        self.f2 = 1500.0
        self.target_f2 = 1500.0
        self.mod_index = MOD_INDEX_MIN
        self.target_mod_index = MOD_INDEX_MIN

        # Active preset (sets ADSR and voice colour).
        self.preset = None
        self.preset_params = PRESETS[DEFAULT_PRESET]
        self.apply_preset(DEFAULT_PRESET)

        # Per-filter state (delay units used by lfilter to keep continuity
        # across blocks).
        self._zi1 = np.zeros(2)
        self._zi2 = np.zeros(2)

    # ---- vision-thread API ----
    def note_on(self, voice_idx, freq):
        self.voices[voice_idx].note_on(freq)

    def note_off(self, voice_idx):
        self.voices[voice_idx].note_off()

    def apply_preset(self, name):
        """Switch timbre: overwrite every voice's ADSR params and colour."""
        p = PRESETS[name]
        for v in self.voices:
            v.amp_env.attack = p["amp_attack"]
            v.amp_env.decay = p["amp_decay"]
            v.amp_env.sustain = p["amp_sustain"]
            v.amp_env.release = p["amp_release"]
            v.mod_env.attack = p["mod_attack"]
            v.mod_env.decay = p["mod_decay"]
            v.mod_env.sustain = p["mod_sustain"]
            v.mod_env.release = p["mod_release"]
        self.preset_params = p
        self.preset = name

    @property
    def any_gate(self):
        return any(v.gate for v in self.voices)

    # ---- audio thread ----
    def callback(self, outdata, frames, _time, status):
        if status:
            print("audio status:", status)

        # Smooth shared params one step per block to remove zipper noise.
        self.f1 += 0.35 * (self.target_f1 - self.f1)
        self.f2 += 0.35 * (self.target_f2 - self.f2)
        self.mod_index += 0.35 * (self.target_mod_index - self.mod_index)

        # 1. Render every voice and sum the FM sources.
        mixed_source = np.zeros(frames, dtype=np.float64)
        preset = self.preset_params
        for v in self.voices:
            mixed_source += v.render(frames, self.mod_index, preset)

        # 2. Two formant band-pass resonators, shared across voices.
        q = preset.get("filter_q", FILTER_Q)
        b1, a1 = iirpeak(self.f1, q, fs=self.sr)
        b2, a2 = iirpeak(self.f2, q, fs=self.sr)
        y1, self._zi1 = lfilter(b1, a1, mixed_source, zi=self._zi1)
        y2, self._zi2 = lfilter(b2, a2, mixed_source, zi=self._zi2)

        # 3. Mix and output (fully linear -- no soft clipper, no waveshaping
        # distortion). MASTER_AMP is kept low so normal play stays under ±1;
        # the np.clip below is just a hard safety net for extreme worst-case
        # spikes and shouldn't trigger in everyday use.
        mixed = (
            preset["f1_mix"] * y1
            + preset["f2_mix"] * y2
            + preset["dry_mix"] * mixed_source
        )
        out = MASTER_AMP * preset["gain"] * mixed
        np.clip(out, -0.99, 0.99, out=out)
        outdata[:, 0] = out.astype(np.float32)


# ===== MediaPipe models =====
# GestureRecognizer gives all three we need: hand landmarks, left/right label
# and a discrete gesture class per hand.
gesture_options = vision.GestureRecognizerOptions(
    base_options=python.BaseOptions(model_asset_path=GESTURE_MODEL),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)
face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
    running_mode=vision.RunningMode.VIDEO,
    output_face_blendshapes=True,
    num_faces=1,
)

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (0, 17),  # palm base
]

# Outer / inner lip ring landmark indices in MediaPipe Face Mesh.
LIPS_OUTER = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    409,
    270,
    269,
    267,
    0,
    37,
    39,
    40,
    185,
]
LIPS_INNER = [
    78,
    95,
    88,
    178,
    87,
    14,
    317,
    402,
    318,
    324,
    308,
    415,
    310,
    311,
    312,
    13,
    82,
    81,
    80,
    191,
]


def finger_curl_ratios(landmarks):
    """Curl ratio for each of the 5 fingers (thumb, index, middle, ring, pinky).

    Four-finger metric: dist(wrist, tip) / dist(wrist, PIP joint). Straight
    fingers stretch the tip beyond the PIP -> ratio > 1; bent fingers fold
    the tip back toward the palm -> ratio < 1.
    Thumb is special: dist(tip 4, palm 9) / dist(thumb-MCP 2, palm 9), since
    the thumb folds across the palm rather than toward the wrist.
    Distance-based metrics keep working under hand rotation.
    """

    def dist(a, b):
        return np.hypot(
            landmarks[a].x - landmarks[b].x, landmarks[a].y - landmarks[b].y
        )

    ratios = []
    base = dist(2, 9)
    ratios.append(dist(4, 9) / base if base > 1e-6 else 1.0)  # thumb
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        d_pip = dist(0, pip)
        ratios.append(dist(0, tip) / d_pip if d_pip > 1e-6 else 1.0)
    return ratios


def draw_hand(frame, landmarks, color):
    """Draw the 21 landmarks of one hand plus the connecting bones."""
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)


def draw_finger_keys(frame, landmarks, finger_down):
    """Mark each fingertip with its note number; bent ones are filled orange."""
    h, w = frame.shape[:2]
    for i, tip in enumerate(FINGER_TIPS):
        x = int(landmarks[tip].x * w)
        y = int(landmarks[tip].y * h)
        color = (0, 140, 255) if finger_down[i] else (200, 200, 200)
        cv2.circle(frame, (x, y), 9, color, -1)
        cv2.putText(
            frame,
            str(i + 1),
            (x - 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (30, 30, 30),
            2,
        )


def draw_face(frame, face_result):
    """Draw the face-mesh dots and outline the lips."""
    h, w = frame.shape[:2]
    for face_landmarks in face_result.face_landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in face_landmarks]
        for x, y in pts:
            cv2.circle(frame, (x, y), 1, (0, 200, 0), -1)
        for idx_list in (LIPS_OUTER, LIPS_INNER):
            poly = np.array([pts[i] for i in idx_list], dtype=np.int32)
            cv2.polylines(frame, [poly], True, (0, 255, 255), 2)


def nearest_vowel(f1, f2):
    """Closest reference vowel to the current (F1, F2). Display only."""
    best, best_d = "?", 1e18
    for name, (rf1, rf2) in VOWEL_REF.items():
        d = ((f1 - rf1) / 600.0) ** 2 + ((f2 - rf2) / 1500.0) ** 2
        if d < best_d:
            best, best_d = name, d
    return best


def main():
    synth = FormantSynth(SAMPLE_RATE)

    # Smoothed normalised control values (one-pole low-pass on raw inputs).
    sm_dist = 0.0  # hand distance -> brightness
    sm_open = 0.0  # jaw open      -> F1
    sm_mouth = 0.5  # smile/pucker  -> F2

    # Finger-keyboard state (one voice per finger).
    finger_down = [False] * 5

    # Preset-switch debounce.
    pending_preset = synth.preset
    pending_preset_frames = 0

    cap = cv2.VideoCapture(0)
    # Keep the camera-driver buffer at 1 so read() always returns the freshest
    # frame instead of replaying stale ones. This is one of the largest
    # invisible sources of perceived latency.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype="float32",
        latency="low",  # ask the driver for the smallest buffer
        callback=synth.callback,
    )

    gesture_rec = vision.GestureRecognizer.create_from_options(gesture_options)
    face_lm = vision.FaceLandmarker.create_from_options(face_options)

    with gesture_rec, face_lm, stream:
        print(
            "Polyphonic finger synth: left hand plays 5 notes "
            "(multiple bent = chord); right hand picks timbre "
            "(palm=pad, fist=key, V=glass, point=drone); "
            "two-hand distance controls brightness. Press q to quit."
        )
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror so labels match the user's view
            timestamp = int(time.time() * 1000)
            rgb = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )

            gesture_result = gesture_rec.recognize_for_video(rgb, timestamp)
            face_result = face_lm.detect_for_video(rgb, timestamp)

            # ---- Split hands: left = keyboard, right = expression / gesture ----
            kb_hand = None  # left hand: 5-finger keyboard
            expr_hand = None  # right hand: distance to kb_hand -> brightness
            expr_gesture = "None"
            for i in range(len(gesture_result.hand_landmarks)):
                label = gesture_result.handedness[i][0].category_name
                lm = gesture_result.hand_landmarks[i]
                if label == "Left":
                    kb_hand = lm
                elif label == "Right":
                    expr_hand = lm
                    gs = gesture_result.gestures[i]
                    if gs:
                        expr_gesture = gs[0].category_name

            # ---- Right-hand gesture -> preset switch (debounced) ----
            target_preset = GESTURE_TO_PRESET.get(expr_gesture)
            if target_preset is None or target_preset == synth.preset:
                pending_preset_frames = 0
                pending_preset = synth.preset
            else:
                if target_preset == pending_preset:
                    pending_preset_frames += 1
                else:
                    pending_preset = target_preset
                    pending_preset_frames = 1
                if pending_preset_frames >= PRESET_DEBOUNCE_FRAMES:
                    synth.apply_preset(pending_preset)
                    pending_preset_frames = 0

            # ---- Left hand: per-finger curl -> per-voice note_on / note_off ----
            cur_ratios = [1.0] * 5
            if kb_hand is not None:
                cur_ratios = finger_curl_ratios(kb_hand)
                for i in range(5):
                    r = cur_ratios[i]
                    if not finger_down[i] and r < CURL_ON[i]:
                        finger_down[i] = True
                        synth.note_on(i, NOTE_FREQS[i])
                    elif finger_down[i] and r > CURL_OFF[i]:
                        finger_down[i] = False
                        synth.note_off(i)
                draw_hand(frame, kb_hand, (255, 0, 255))
                draw_finger_keys(frame, kb_hand, finger_down)
            else:
                # Lost the left hand: release everything (no stuck notes).
                for i in range(5):
                    if finger_down[i]:
                        finger_down[i] = False
                        synth.note_off(i)

            # ---- Two-hand horizontal distance -> brightness (MOD_INDEX) ----
            hand_dist = 0.0
            if kb_hand is not None and expr_hand is not None:
                hand_dist = abs(kb_hand[0].x - expr_hand[0].x)
                dnorm = float(
                    np.clip((hand_dist - DIST_MIN) / (DIST_MAX - DIST_MIN), 0.0, 1.0)
                )
                sm_dist += 0.4 * (dnorm - sm_dist)
                synth.target_mod_index = MOD_INDEX_MIN + sm_dist * (
                    MOD_INDEX_MAX - MOD_INDEX_MIN
                )
                draw_hand(frame, expr_hand, (255, 128, 0))
                # Draw a wrist-to-wrist line as a visible "distance" cue.
                h, w = frame.shape[:2]
                kw = (int(kb_hand[0].x * w), int(kb_hand[0].y * h))
                ew = (int(expr_hand[0].x * w), int(expr_hand[0].y * h))
                cv2.line(frame, kw, ew, (0, 255, 255), 2)
            elif expr_hand is not None:
                # Right hand alone: still draw it (gestures still work).
                draw_hand(frame, expr_hand, (255, 128, 0))

            # ---- Mouth shape -> formants F1 / F2 ----
            vowel = "?"
            if face_result.face_landmarks and face_result.face_blendshapes:
                bs = {c.category_name: c.score for c in face_result.face_blendshapes[0]}
                jaw_open = bs.get("jawOpen", 0.0)
                smile = (
                    bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)
                ) / 2.0
                pucker = bs.get("mouthPucker", 0.0)

                open_amt = float(np.clip(jaw_open, 0.0, 1.0))
                mouth_amt = float(np.clip((smile - 2.5 * pucker + 1.0) / 2.0, 0.0, 1.0))

                sm_open += 0.4 * (open_amt - sm_open)
                sm_mouth += 0.4 * (mouth_amt - sm_mouth)

                synth.target_f1 = F1_MIN + sm_open * (F1_MAX - F1_MIN)
                synth.target_f2 = F2_MIN + sm_mouth * (F2_MAX - F2_MIN)
                vowel = nearest_vowel(synth.target_f1, synth.target_f2)
                draw_face(frame, face_result)

            # ---- Terminal log ----
            fstr = "".join(str(i + 1) if d else "-" for i, d in enumerate(finger_down))
            rstr = " ".join(f"{r:.2f}" for r in cur_ratios)
            chord = [NOTE_FREQS[i] for i in range(5) if finger_down[i]]
            chord_str = "+".join(f"{f:.0f}" for f in chord) if chord else "-"
            gate_str = "ON " if synth.any_gate else "off"
            print(
                f"preset={synth.preset:5s}({expr_gesture})  "
                f"curl[{rstr}]  fingers[{fstr}]  "
                f"chord=[{chord_str}]Hz  "
                f"bright={synth.target_mod_index:4.1f}(dist={hand_dist:.2f})  "
                f"gate={gate_str}  /{vowel}/"
            )

            # ---- On-screen overlay ----
            preset_color = PRESET_COLORS.get(synth.preset, (200, 200, 200))
            cv2.putText(
                frame,
                f"preset {synth.preset}  fingers [{fstr}]  "
                f"bright={synth.target_mod_index:.1f}  /{vowel}/",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                preset_color,
                2,
            )

            cv2.imshow("DT2213 - Polyphonic Finger Synth", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
