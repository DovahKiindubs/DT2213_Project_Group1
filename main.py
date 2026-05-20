"""
DT2213 项目：左手 5 指键盘 + 右手手势切换音色的 FM Formant 合成器。

FM 结构：每根手指 = 一个 voice（FM 振荡器 A 载波 + 振荡器 B 调制器 + 自己的 ADSR）
所有 voice 求和后过两个共振峰带通（嘴形 -> F1/F2），形成一个"会唱元音的合唱"。

控制方式：
  左手 5 根手指          -> 各对应一个音符（拇指最低 ~ 小指最高）
  弯下手指 / 伸直手指    -> 该音 note-on / note-off（弯着就一直响）
  同时弯下多根          -> 几个音叠起来 = 和弦（多 voice 同时发声）
  右手 ✋张开 / ✊握拳   -> 切换音色：pad（氛围）/ key（清脆）
  两手水平距离          -> 音色明暗（调制深度 MOD_INDEX）
  下巴张开    (Face)    -> 共振峰 F1
  咧嘴/圆唇   (Face)    -> 共振峰 F2

画面已做镜像，符合"照镜子"直觉。
按 q 退出。
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

# 模型文件与本脚本放在同一目录，按绝对路径加载（不依赖运行时的工作目录）
HERE = os.path.dirname(os.path.abspath(__file__))
GESTURE_MODEL = os.path.join(HERE, "gesture_recognizer.task")
FACE_MODEL = os.path.join(HERE, "face_landmarker.task")

# ===== 音频参数 =====
SAMPLE_RATE = 44100
BLOCK_SIZE = 512
MASTER_AMP = 0.4  # 总音量上限

# ===== FM 全局参数 =====
# 调制深度 MOD_INDEX：由两手水平距离控制（音色明暗），区间如下
MOD_INDEX_MIN, MOD_INDEX_MAX = 0.0, 15.0

# ===== 音色预设（pad / key）=====
# 切换预设时一次性改写所有 voice 的 ADSR 参数和 mod_ratio。
# attack/decay/release 单位是秒；sustain 是 0~1 电平；mod_ratio = 调制器 / 载波。
PRESETS = {
    # pad: 慢起、长延、绵长——氛围合唱
    "pad": {
        "amp_attack":  0.50, "amp_decay":  1.50, "amp_sustain": 0.70, "amp_release": 2.50,
        "mod_attack":  1.00, "mod_decay":  0.60, "mod_sustain": 0.80, "mod_release": 2.00,
        "mod_ratio":   1.0,
    },
    # key: 快起、快衰、短尾——清脆敲击，像电钢/钟琴
    "key": {
        "amp_attack":  0.005, "amp_decay": 0.55, "amp_sustain": 0.10, "amp_release": 0.35,
        "mod_attack":  0.005, "mod_decay": 0.35, "mod_sustain": 0.00, "mod_release": 0.30,
        "mod_ratio":   2.0,
    },
}
DEFAULT_PRESET = "pad"

# 右手手势 -> 预设名（其它手势保持当前预设不变）
GESTURE_TO_PRESET = {
    "Open_Palm":   "pad",   # ✋ 张开 -> 氛围
    "Closed_Fist": "key",   # ✊ 握拳 -> 清脆
}
PRESET_DEBOUNCE_FRAMES = 5  # 同一手势连续 N 帧才切换预设，防误切

# ===== 手指键盘 =====
PENTATONIC_STEPS = [0, 2, 4, 7, 9]
SCALE_ROOT_HZ = 65.41   # C2
KEYBOARD_OCTAVE = 1     # 5 个音落在第几个八度（0 起）

# 手指弯曲判定（带迟滞，防抖动）：弯曲比例低于 CURL_ON 判为弯下、
# 高于 CURL_OFF 判为伸直。每根手指独立——顺序：拇/食/中/无名/小。
# 拇指在手掌放松时本来就略弯，阈值要明显更低，需刻意大幅弯曲才触发。
CURL_ON = [0.70, 0.95, 0.95, 0.95, 0.95]
CURL_OFF = [0.75, 1.00, 1.00, 1.00, 1.00]

# 共振峰可达范围（Hz）—— 由嘴形控制
F1_MIN, F1_MAX = 200.0, 1600.0
F2_MIN, F2_MAX = 500.0, 3200.0

# 两个共振峰谐振器的 Q 值（固定）
FILTER_Q = 4.0

# 两手手腕的水平距离映射到调制深度（音色明暗）的区间。
# 两手靠拢 -> 暗，张开 -> 亮；先看终端打印的 dist 实际值再校准
DIST_MIN, DIST_MAX = 0.15, 0.75

# 5 个参考元音的 (F1, F2)，仅用于在终端/画面显示"猜到的元音"
VOWEL_REF = {
    "i": (240, 2400),
    "e": (390, 2300),
    "a": (860, 1600),
    "o": (360, 630),
    "u": (240, 560),
}


def note_freq(degree, octave):
    """五声音阶第 degree 级（1~5）、第 octave 个八度（0 起）-> 频率（Hz）。"""
    step = PENTATONIC_STEPS[degree - 1]
    return SCALE_ROOT_HZ * 2.0 ** (octave + step / 12.0)


# 5 根手指对应的音符频率：拇指(0)最低 ~ 小指(4)最高
NOTE_FREQS = [note_freq(d, KEYBOARD_OCTAVE) for d in range(1, 6)]

# 5 根手指的指尖关键点（拇/食/中/无名/小）
FINGER_TIPS = [4, 8, 12, 16, 20]


class ADSR:
    """线性 ADSR 包络发生器。每次 process() 按 gate 状态生成一段包络。

    attack/decay/sustain/release 可以在运行时被改（apply_preset 切音色时会改）。
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
    """单个音符：FM 振荡器（A 载波 + B 调制器）+ 自己的 ADSR + 自己的相位。

    note_on/note_off 由视觉线程调用；render 由音频线程在每个块调用一次。
    """

    def __init__(self, sample_rate):
        self.sr = sample_rate
        self.freq = 220.0
        self.gate = False
        self._phase_c = 0.0
        self._phase_m = 0.0
        # ADSR 起始用 pad 的默认值；apply_preset 会立刻覆盖
        self.amp_env = ADSR(sample_rate, 0.5, 1.5, 0.7, 2.5)
        self.mod_env = ADSR(sample_rate, 1.0, 0.6, 0.8, 2.0)

    def note_on(self, freq):
        self.freq = freq
        self.gate = True

    def note_off(self):
        self.gate = False

    def render(self, frames, mod_index, mod_ratio):
        """渲染本块的 FM 信号 × amp_env；返回长度 frames 的 numpy。"""
        amp_env = self.amp_env.process(frames, self.gate)
        mod_env = self.mod_env.process(frames, self.gate)
        # 本块内 voice 的频率不变 -> 用简单累加即可（无需 cumsum）
        n = np.arange(1, frames + 1)
        dphi_c = 2.0 * np.pi * self.freq / self.sr
        dphi_m = 2.0 * np.pi * self.freq * mod_ratio / self.sr
        phase_c = self._phase_c + dphi_c * n
        phase_m = self._phase_m + dphi_m * n
        self._phase_c = phase_c[-1] % (2.0 * np.pi)
        self._phase_m = phase_m[-1] % (2.0 * np.pi)
        source = np.sin(phase_c + (mod_index * mod_env) * np.sin(phase_m))
        return source * amp_env


class FormantSynth:
    """N 个 Voice 同时发声 -> 求和 -> 两个共振峰带通 -> 输出。

    多 voice 同时 gate=True 即和弦；同一 voice 仅对应一根手指。
    所有 voice 共享 mod_index（亮度）、mod_ratio（音色）、F1/F2（元音）。
    """

    def __init__(self, sample_rate, num_voices=5):
        self.sr = sample_rate
        self.voices = [Voice(sample_rate) for _ in range(num_voices)]

        # 全局共振峰 + 调制深度（视觉线程写 target_*，callback 平滑跟随）
        self.f1 = 600.0; self.target_f1 = 600.0
        self.f2 = 1500.0; self.target_f2 = 1500.0
        self.mod_index = MOD_INDEX_MIN; self.target_mod_index = MOD_INDEX_MIN

        # 音色预设（pad / key）
        self.preset = None
        self.mod_ratio = 1.0
        self.apply_preset(DEFAULT_PRESET)

        # 共振峰滤波器状态（lfilter 的延迟单元）
        self._zi1 = np.zeros(2)
        self._zi2 = np.zeros(2)

    # ---- 视觉线程接口 ----
    def note_on(self, voice_idx, freq):
        self.voices[voice_idx].note_on(freq)

    def note_off(self, voice_idx):
        self.voices[voice_idx].note_off()

    def apply_preset(self, name):
        """切换音色：覆盖所有 voice 的 ADSR 参数和 mod_ratio。"""
        p = PRESETS[name]
        for v in self.voices:
            v.amp_env.attack  = p["amp_attack"]
            v.amp_env.decay   = p["amp_decay"]
            v.amp_env.sustain = p["amp_sustain"]
            v.amp_env.release = p["amp_release"]
            v.mod_env.attack  = p["mod_attack"]
            v.mod_env.decay   = p["mod_decay"]
            v.mod_env.sustain = p["mod_sustain"]
            v.mod_env.release = p["mod_release"]
        self.mod_ratio = p["mod_ratio"]
        self.preset = name

    @property
    def any_gate(self):
        return any(v.gate for v in self.voices)

    # ---- 音频线程 ----
    def callback(self, outdata, frames, _time, status):
        if status:
            print("audio status:", status)

        # 全局参数每块向目标平滑一步，消除跳变
        self.f1 += 0.35 * (self.target_f1 - self.f1)
        self.f2 += 0.35 * (self.target_f2 - self.f2)
        self.mod_index += 0.35 * (self.target_mod_index - self.mod_index)

        # ---- 1. 渲染并求和所有 voice 的 FM 声源 ----
        mixed_source = np.zeros(frames, dtype=np.float64)
        for v in self.voices:
            mixed_source += v.render(frames, self.mod_index, self.mod_ratio)

        # ---- 2. 两个共振峰带通谐振器（公用）----
        b1, a1 = iirpeak(self.f1, FILTER_Q, fs=self.sr)
        b2, a2 = iirpeak(self.f2, FILTER_Q, fs=self.sr)
        y1, self._zi1 = lfilter(b1, a1, mixed_source, zi=self._zi1)
        y2, self._zi2 = lfilter(b2, a2, mixed_source, zi=self._zi2)

        # ---- 3. 混合 + 软削波 ----
        mixed = 0.5 * y1 + 0.5 * y2
        out = MASTER_AMP * np.tanh(1.5 * mixed)
        outdata[:, 0] = out.astype(np.float32)


# ===== MediaPipe 模型 =====
# GestureRecognizer 同时给出：手部 21 关键点 + 左右手标签 + 手势分类
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
    (0, 1), (1, 2), (2, 3), (3, 4),         # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),         # 食指
    (5, 9), (9, 10), (10, 11), (11, 12),    # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),  # 无名指
    (13, 17), (17, 18), (18, 19), (19, 20), # 小指
    (0, 17),                                # 掌根
]

LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
              291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
              308, 415, 310, 311, 312, 13, 82, 81, 80, 191]


def finger_curl_ratios(landmarks):
    """5 根手指的弯曲比例（拇/食/中/无名/小）。直伸 >1、弯曲 <1。"""
    def dist(a, b):
        return np.hypot(landmarks[a].x - landmarks[b].x,
                        landmarks[a].y - landmarks[b].y)

    ratios = []
    base = dist(2, 9)
    ratios.append(dist(4, 9) / base if base > 1e-6 else 1.0)  # 拇指
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        d_pip = dist(0, pip)
        ratios.append(dist(0, tip) / d_pip if d_pip > 1e-6 else 1.0)
    return ratios


def draw_hand(frame, landmarks, color):
    h, w = frame.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)


def draw_finger_keys(frame, landmarks, finger_down):
    """在键盘手 5 个指尖标出音符序号；弯下的手指 = 橙色实心。"""
    h, w = frame.shape[:2]
    for i, tip in enumerate(FINGER_TIPS):
        x = int(landmarks[tip].x * w)
        y = int(landmarks[tip].y * h)
        color = (0, 140, 255) if finger_down[i] else (200, 200, 200)
        cv2.circle(frame, (x, y), 9, color, -1)
        cv2.putText(frame, str(i + 1), (x - 5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 2)


def draw_face(frame, face_result):
    h, w = frame.shape[:2]
    for face_landmarks in face_result.face_landmarks:
        pts = [(int(p.x * w), int(p.y * h)) for p in face_landmarks]
        for x, y in pts:
            cv2.circle(frame, (x, y), 1, (0, 200, 0), -1)
        for idx_list in (LIPS_OUTER, LIPS_INNER):
            poly = np.array([pts[i] for i in idx_list], dtype=np.int32)
            cv2.polylines(frame, [poly], True, (0, 255, 255), 2)


def nearest_vowel(f1, f2):
    best, best_d = "?", 1e18
    for name, (rf1, rf2) in VOWEL_REF.items():
        d = ((f1 - rf1) / 600.0) ** 2 + ((f2 - rf2) / 1500.0) ** 2
        if d < best_d:
            best, best_d = name, d
    return best


def main():
    synth = FormantSynth(SAMPLE_RATE)

    # 平滑后的归一化控制量
    sm_dist = 0.0    # 两手距离 -> 音色明暗
    sm_open = 0.0    # 下巴张开 -> F1
    sm_mouth = 0.5   # 咧嘴/圆唇 -> F2

    # 手指键盘状态（每根手指对应一个 voice）
    finger_down = [False] * 5

    # 预设切换去抖
    pending_preset = synth.preset
    pending_preset_frames = 0

    cap = cv2.VideoCapture(0)
    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        dtype="float32",
        callback=synth.callback,
    )

    gesture_rec = vision.GestureRecognizer.create_from_options(gesture_options)
    face_lm = vision.FaceLandmarker.create_from_options(face_options)

    with gesture_rec, face_lm, stream:
        print("和弦合成器：左手 5 指弹音（同时弯多根 = 和弦），"
              "右手 ✋=pad ✊=key，两手距离控音色。按 q 退出。")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # 镜像
            timestamp = int(time.time() * 1000)
            rgb = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )

            gesture_result = gesture_rec.recognize_for_video(rgb, timestamp)
            face_result = face_lm.detect_for_video(rgb, timestamp)

            # ---- 分配两只手：左手 = 键盘手，右手 = 表达/手势手 ----
            kb_hand = None    # 左手：5 指键盘
            expr_hand = None  # 右手：与左手的距离 -> 音色明暗
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

            # ---- 右手手势 -> 切换音色（带去抖）----
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

            # ---- 左手：每根手指弯曲 -> 对应 voice，弯着就一直响 ----
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
                # 键盘手丢失：所有手指释放（不留挂着的音）
                for i in range(5):
                    if finger_down[i]:
                        finger_down[i] = False
                        synth.note_off(i)

            # ---- 两手水平距离 -> 音色明暗（调制深度 MOD_INDEX）----
            hand_dist = 0.0
            if kb_hand is not None and expr_hand is not None:
                hand_dist = abs(kb_hand[0].x - expr_hand[0].x)
                dnorm = float(np.clip(
                    (hand_dist - DIST_MIN) / (DIST_MAX - DIST_MIN), 0.0, 1.0))
                sm_dist += 0.4 * (dnorm - sm_dist)
                synth.target_mod_index = (
                    MOD_INDEX_MIN + sm_dist * (MOD_INDEX_MAX - MOD_INDEX_MIN)
                )
                draw_hand(frame, expr_hand, (255, 128, 0))
                h, w = frame.shape[:2]
                kw = (int(kb_hand[0].x * w), int(kb_hand[0].y * h))
                ew = (int(expr_hand[0].x * w), int(expr_hand[0].y * h))
                cv2.line(frame, kw, ew, (0, 255, 255), 2)
            elif expr_hand is not None:
                # 没看见左手但有右手：也画出右手（手势识别还是有效的）
                draw_hand(frame, expr_hand, (255, 128, 0))

            # ---- 嘴形 -> 共振峰 F1 / F2 ----
            vowel = "?"
            if face_result.face_landmarks and face_result.face_blendshapes:
                bs = {c.category_name: c.score
                      for c in face_result.face_blendshapes[0]}
                jaw_open = bs.get("jawOpen", 0.0)
                smile = (bs.get("mouthSmileLeft", 0.0)
                         + bs.get("mouthSmileRight", 0.0)) / 2.0
                pucker = bs.get("mouthPucker", 0.0)

                open_amt = float(np.clip(jaw_open, 0.0, 1.0))
                mouth_amt = float(np.clip((smile - pucker + 1.0) / 2.0, 0.0, 1.0))

                sm_open += 0.4 * (open_amt - sm_open)
                sm_mouth += 0.4 * (mouth_amt - sm_mouth)

                synth.target_f1 = F1_MIN + sm_open * (F1_MAX - F1_MIN)
                synth.target_f2 = F2_MIN + sm_mouth * (F2_MAX - F2_MIN)
                vowel = nearest_vowel(synth.target_f1, synth.target_f2)
                draw_face(frame, face_result)

            # ---- 终端输出 ----
            fstr = "".join(str(i + 1) if d else "-"
                           for i, d in enumerate(finger_down))
            rstr = " ".join(f"{r:.2f}" for r in cur_ratios)
            chord = [NOTE_FREQS[i] for i in range(5) if finger_down[i]]
            chord_str = "+".join(f"{f:.0f}" for f in chord) if chord else "-"
            gate_str = "ON " if synth.any_gate else "off"
            print(
                f"preset={synth.preset:3s}({expr_gesture})  "
                f"curl[{rstr}]  fingers[{fstr}]  "
                f"chord=[{chord_str}]Hz  "
                f"bright={synth.target_mod_index:4.1f}(dist={hand_dist:.2f})  "
                f"gate={gate_str}  /{vowel}/"
            )

            # ---- 画面叠加文字 ----
            preset_color = (0, 255, 180) if synth.preset == "pad" else (180, 220, 255)
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
