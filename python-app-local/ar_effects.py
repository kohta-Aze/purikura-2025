from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageSequence

from ar.crown_tracker import CrownTracker

try:
    import mediapipe as mp  # type: ignore
except ImportError as exc:  # pragma: no cover - handled at runtime
    mp = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Sample asset generation (used in environments where binary assets are absent)
# ---------------------------------------------------------------------------


def _default_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (256, 256), (255, 170, 200, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 96, 224, 208), fill=(255, 120, 160, 200), outline=(255, 255, 255, 255), width=4)
    font = ImageFont.load_default()
    draw.text((48, 128), "AR", font=font, fill=(255, 255, 255, 255))
    image.save(path)


def _crown_sprite(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (420, 220), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    base_y = 170
    crown_points = [
        (20, base_y),
        (90, 80),
        (150, base_y - 40),
        (210, 50),
        (270, base_y - 40),
        (330, 80),
        (400, base_y),
    ]
    draw.polygon(crown_points, fill=(255, 210, 32, 235), outline=(255, 240, 120, 255))
    jewel_colors = [(120, 190, 255, 240), (255, 120, 150, 240), (180, 255, 150, 240)]
    jewel_centers = [(90, 120), (210, 90), (330, 120)]
    for (cx, cy), color in zip(jewel_centers, jewel_colors):
        draw.ellipse((cx - 24, cy - 24, cx + 24, cy + 24), fill=color, outline=(255, 255, 255, 255), width=4)
    draw.rectangle((40, base_y, 380, base_y + 32), fill=(255, 204, 0, 235), outline=(255, 240, 150, 255), width=4)
    image = image.filter(ImageFilter.GaussianBlur(0.5))
    image.save(path)


def _fairy_glow_sprite(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    size = (280, 260)
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    center = (size[0] // 2, size[1] // 2)
    radii = [120, 95, 70, 48]
    alphas = [30, 60, 110, 160]
    for radius, alpha in zip(radii, alphas):
        bbox = [
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ]
        draw.ellipse(bbox, fill=(200, 225, 255, alpha))
    sparkle_color = (255, 255, 240, 200)
    for offset in range(-60, 61, 30):
        draw.ellipse(
            (center[0] + offset - 10, center[1] - 10, center[0] + offset + 10, center[1] + 10),
            fill=sparkle_color,
        )
    image = image.filter(ImageFilter.GaussianBlur(2))
    image.save(path)


def _cheek_heart_sprite(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (220, 200), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    heart_color = (255, 128, 170, 220)
    highlight = (255, 210, 230, 220)
    def heart(center: tuple[int, int], size: int) -> None:
        x, y = center
        w = size
        h = int(size * 0.9)
        top = y - h // 2
        draw.pieslice((x - w, top, x, top + h), 180, 360, fill=heart_color)
        draw.pieslice((x, top, x + w, top + h), 180, 360, fill=heart_color)
        draw.polygon((x - w, y, x, y + h, x + w, y), fill=heart_color)
        draw.ellipse((x - w // 3, y - h // 3, x - w // 10, y - h // 6), fill=highlight)
    heart((110, 110), 70)
    image = image.filter(ImageFilter.GaussianBlur(0.6))
    image.save(path)


def _uniform_sprite(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (480, 520), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    body_color = (40, 70, 130, 235)
    collar_color = (240, 240, 250, 255)
    trim_color = (255, 200, 80, 255)
    torso = [(80, 60), (400, 60), (450, 360), (420, 480), (60, 480), (30, 360)]
    draw.polygon(torso, fill=body_color)
    collar = [(120, 60), (240, 220), (320, 220), (440, 60)]
    draw.polygon(collar, fill=collar_color)
    draw.line([(240, 220), (240, 480)], fill=trim_color, width=12)
    draw.line([(320, 220), (320, 480)], fill=trim_color, width=12)
    draw.line([(120, 60), (240, 220), (320, 220), (440, 60)], fill=trim_color, width=8)
    image = image.filter(ImageFilter.GaussianBlur(0.8))
    image.save(path)


def _reitaku_emblem_sprite(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    size = (220, 220)
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    center = (size[0] // 2, size[1] // 2)
    draw.ellipse((20, 20, size[0] - 20, size[1] - 20), fill=(0, 90, 60, 240), outline=(255, 215, 0, 255), width=6)
    draw.ellipse((60, 60, size[0] - 60, size[1] - 60), outline=(255, 215, 0, 220), width=6)
    draw.line((center[0], 60, center[0], size[1] - 60), fill=(255, 215, 0, 255), width=8)
    draw.line((60, center[1], size[0] - 60, center[1]), fill=(255, 215, 0, 255), width=8)
    draw.ellipse((center[0] - 12, center[1] - 12, center[0] + 12, center[1] + 12), fill=(255, 255, 255, 255))
    image = image.filter(ImageFilter.GaussianBlur(0.5))
    image.save(path)


def _bg_sakura_sprite(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    size = (960, 720)
    gradient = Image.new("RGBA", size)
    top_color = np.array([255, 220, 235, 255], dtype=np.float32)
    bottom_color = np.array([240, 180, 220, 255], dtype=np.float32)
    for y in range(size[1]):
        ratio = y / max(1, size[1] - 1)
        row = (top_color * (1 - ratio) + bottom_color * ratio).astype(np.uint8)
        color = tuple(int(channel) for channel in row)
        gradient.paste(color, box=(0, y, size[0], y + 1))
    draw = ImageDraw.Draw(gradient)
    petal_color = (255, 245, 250, 160)
    for x in range(80, size[0], 160):
        for y in range(60, size[1], 180):
            bbox = (x - 50, y - 20, x + 50, y + 60)
            draw.ellipse(bbox, fill=petal_color)
    blur = gradient.filter(ImageFilter.GaussianBlur(6))
    blur.save(path)


_SAMPLE_SPRITE_FACTORIES: dict[str, Callable[[Path], None]] = {
    "crown_basic": _crown_sprite,
    "crown": _crown_sprite,
    "fairy_glow": _fairy_glow_sprite,
    "fairy_glow.png": _fairy_glow_sprite,
    "cheek_heart": _cheek_heart_sprite,
    "cheek_heart.png": _cheek_heart_sprite,
    "uniform_top": _uniform_sprite,
    "uniform": _uniform_sprite,
    "reitaku_emblem_hand": _reitaku_emblem_sprite,
    "reitaku_emblem": _reitaku_emblem_sprite,
    "bg_sakura": _bg_sakura_sprite,
    "bg_sakura.png": _bg_sakura_sprite,
}


def ensure_sample_sprite(effect_name: str, path: Path) -> None:
    sprite_path = Path(path)
    if sprite_path.exists():
        return
    generator = _SAMPLE_SPRITE_FACTORIES.get(effect_name) or _SAMPLE_SPRITE_FACTORIES.get(sprite_path.stem)
    if generator is None:
        generator = _default_placeholder
    generator(sprite_path)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SpriteFrame:
    """Holds one RGBA frame (for static or animated sprites)."""

    image_rgba: Image.Image
    duration: float

    @property
    def image_bgra(self) -> np.ndarray:
        arr = np.array(self.image_rgba, dtype=np.uint8)
        if arr.shape[-1] != 4:
            arr = np.dstack((arr, np.full(arr.shape[:2], 255, dtype=np.uint8)))
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)


@dataclass
class Overlay:
    """Overlay instruction for AR composition."""

    image: np.ndarray
    x: int
    y: int
    z_index: int
    mask_with_segmentation: bool
    alpha_multiplier: float = 1.0


# ---------------------------------------------------------------------------
# Mediapipe-based AR engine
# ---------------------------------------------------------------------------


class AREngine:
    """Runs landmark/segmentation estimation asynchronously."""

    FACE_LANDMARKS = {
        "FOREHEAD_CENTER": 10,
        "LEFT_EAR": 234,
        "RIGHT_EAR": 454,
        "LEFT_EYE_OUTER": 33,
        "RIGHT_EYE_OUTER": 263,
        "LEFT_EYE_INNER": 133,
        "RIGHT_EYE_INNER": 362,
        "NOSE_TIP": 1,
        "CHIN": 152,
        "LEFT_CHEEK": 123,
        "RIGHT_CHEEK": 352,
    }

    POSE_LANDMARKS = {
        "LEFT_SHOULDER": 11,
        "RIGHT_SHOULDER": 12,
        "LEFT_HIP": 23,
        "RIGHT_HIP": 24,
        "LEFT_EAR": 7,
        "RIGHT_EAR": 8,
        "NOSE": 0,
    }

    HAND_LANDMARKS = {
        "WRIST": 0,
        "THUMB_CMC": 1,
        "THUMB_MCP": 2,
        "THUMB_IP": 3,
        "THUMB_TIP": 4,
        "INDEX_MCP": 5,
        "INDEX_PIP": 6,
        "INDEX_DIP": 7,
        "INDEX_TIP": 8,
        "MIDDLE_MCP": 9,
        "MIDDLE_PIP": 10,
        "MIDDLE_DIP": 11,
        "MIDDLE_TIP": 12,
        "RING_MCP": 13,
        "RING_PIP": 14,
        "RING_DIP": 15,
        "RING_TIP": 16,
        "PINKY_MCP": 17,
        "PINKY_PIP": 18,
        "PINKY_DIP": 19,
        "PINKY_TIP": 20,
    }

    def __init__(self, detection_size: tuple[int, int] = (640, 480), ema_alpha: float = 0.3):
        if mp is None:
            raise RuntimeError(
                "mediapipe がインストールされていません。`pip install mediapipe` を実行してください。"
            ) from _IMPORT_ERROR
        self.detection_size = detection_size
        self.ema_alpha = ema_alpha
        self._frame_lock = threading.Condition()
        self._inference_lock = threading.Lock()
        self._latest_frame: tuple[np.ndarray, float] | None = None
        self._last_state: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._segmentation_mask: np.ndarray | None = None
        self._ema_store: dict[str, dict[str, np.ndarray]] = {}
        self._face_state: dict[str, Any] | None = None
        self._pose_state: dict[str, Any] | None = None
        self._hand_states: dict[str, dict[str, Any]] = {}
        self._retention = 0.6  # seconds
        self._running = True
        self._worker = threading.Thread(target=self._loop, daemon=True)

        mp_face_mesh = mp.solutions.face_mesh
        mp_hands = mp.solutions.hands
        mp_pose = mp.solutions.pose
        mp_seg = mp.solutions.selfie_segmentation

        self._face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._hands = mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._segmentation = mp_seg.SelfieSegmentation(model_selection=1)

        self._worker.start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_frame(self, frame: np.ndarray, timestamp: float | None = None) -> None:
        if frame is None:
            return
        if timestamp is None:
            timestamp = time.time()
        with self._frame_lock:
            self._latest_frame = (frame.copy(), timestamp)
            self._frame_lock.notify()

    def get_state(self) -> dict[str, Any] | None:
        return self._last_state

    def pop_last_error(self) -> str | None:
        err = self._last_error
        self._last_error = None
        return err

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            with self._frame_lock:
                if self._latest_frame is None:
                    self._frame_lock.wait(timeout=0.05)
                    continue
                frame, ts = self._latest_frame
                self._latest_frame = None
            try:
                with self._inference_lock:
                    state = self._process_frame(frame, ts)
                self._last_state = state
            except Exception as exc:  # pragma: no cover - runtime diagnostic
                self._last_error = f"AR推論でエラーが発生しました: {exc}"

    # ------------------------------------------------------------------
    # Processing helpers
    # ------------------------------------------------------------------

    def process_static_frame(self, frame: np.ndarray, timestamp: float | None = None) -> dict[str, Any] | None:
        """Process a single still frame synchronously and return the AR state."""

        if frame is None:
            return self._last_state
        if timestamp is None:
            timestamp = time.time()
        try:
            with self._inference_lock:
                state = self._process_frame(frame, timestamp)
        except Exception as exc:  # pragma: no cover - runtime diagnostic
            self._last_error = f"AR推論でエラーが発生しました: {exc}"
            return self._last_state
        self._last_state = state
        return state

    def _process_frame(self, frame: np.ndarray, timestamp: float) -> dict[str, Any]:
        height, width = frame.shape[:2]
        resized = cv2.resize(frame, self.detection_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        face_result = self._face_mesh.process(rgb)
        pose_result = self._pose.process(rgb)
        hands_result = self._hands.process(rgb)
        seg_result = self._segmentation.process(rgb)

        face_state = self._handle_face(face_result, width, height, timestamp)
        pose_state = self._handle_pose(pose_result, width, height, timestamp)
        hands_state, gestures = self._handle_hands(hands_result, width, height, timestamp)
        segmentation = self._handle_segmentation(seg_result, width, height)

        state = {
            "frame_size": (height, width),
            "face": face_state,
            "pose": pose_state,
            "hands": hands_state,
            "segmentation": segmentation,
            "gestures": gestures,
            "timestamp": timestamp,
        }
        return state

    def _handle_face(self, result, width: int, height: int, timestamp: float) -> dict[str, Any] | None:
        landmarks: dict[str, np.ndarray] | None = None
        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0]
            coords: dict[str, np.ndarray] = {}
            for name, idx in self.FACE_LANDMARKS.items():
                pt = lm.landmark[idx]
                coords[name] = np.array([pt.x * width, pt.y * height, pt.z * width], dtype=np.float32)
            landmarks = self._smooth_group("face", coords)
            self._face_state = {
                "landmarks": landmarks,
                "metrics": self._compute_face_metrics(landmarks),
                "last_seen": timestamp,
            }
        elif self._face_state and timestamp - self._face_state.get("last_seen", 0) > self._retention:
            self._face_state = None
        return self._face_state

    def _handle_pose(self, result, width: int, height: int, timestamp: float) -> dict[str, Any] | None:
        landmarks: dict[str, np.ndarray] | None = None
        if result.pose_landmarks:
            coords: dict[str, np.ndarray] = {}
            for name, idx in self.POSE_LANDMARKS.items():
                pt = result.pose_landmarks.landmark[idx]
                coords[name] = np.array([pt.x * width, pt.y * height, pt.z * width], dtype=np.float32)
            landmarks = self._smooth_group("pose", coords)
            self._pose_state = {
                "landmarks": landmarks,
                "metrics": self._compute_pose_metrics(landmarks),
                "last_seen": timestamp,
            }
        elif self._pose_state and timestamp - self._pose_state.get("last_seen", 0) > self._retention:
            self._pose_state = None
        return self._pose_state

    def _handle_hands(self, result, width: int, height: int, timestamp: float) -> tuple[list[dict[str, Any]], set[str]]:
        gestures: set[str] = set()
        if result.multi_hand_landmarks:
            handedness = result.multi_handedness or []
            for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                label = "Unknown"
                score = 0.0
                if idx < len(handedness):
                    label = handedness[idx].classification[0].label
                    score = handedness[idx].classification[0].score
                coords: dict[str, np.ndarray] = {}
                for name, lm_index in self.HAND_LANDMARKS.items():
                    pt = hand_landmarks.landmark[lm_index]
                    coords[name] = np.array([pt.x * width, pt.y * height, pt.z * width], dtype=np.float32)
                palm_center = np.mean(
                    [
                        coords.get("WRIST"),
                        coords.get("INDEX_MCP"),
                        coords.get("MIDDLE_MCP"),
                        coords.get("RING_MCP"),
                        coords.get("PINKY_MCP"),
                    ],
                    axis=0,
                )
                coords["PALM_CENTER"] = palm_center
                key = f"hand_{label}_{idx}"
                smoothed = self._smooth_group(key, coords)
                metrics = self._compute_hand_metrics(smoothed)
                rotation = math.degrees(
                    math.atan2(
                        smoothed["INDEX_MCP"][1] - smoothed["PINKY_MCP"][1],
                        smoothed["INDEX_MCP"][0] - smoothed["PINKY_MCP"][0],
                    )
                )
                hand_state = {
                    "landmarks": smoothed,
                    "metrics": metrics,
                    "handedness": label,
                    "confidence": score,
                    "rotation": rotation,
                    "last_seen": timestamp,
                }
                gesture = self._detect_bowl_gesture(hand_state)
                if gesture:
                    hand_state["gesture"] = gesture
                    gestures.add(gesture)
                    gestures.add("bowl_hand")
                self._hand_states[key] = hand_state
        # Remove stale hands
        stale_keys = [
            key
            for key, info in self._hand_states.items()
            if timestamp - info.get("last_seen", 0) > self._retention
        ]
        for key in stale_keys:
            self._hand_states.pop(key, None)
        hands = list(self._hand_states.values())
        return hands, gestures

    def _handle_segmentation(self, result, width: int, height: int) -> np.ndarray | None:
        if result.segmentation_mask is None:
            return self._segmentation_mask
        mask = cv2.resize(result.segmentation_mask, (width, height), interpolation=cv2.INTER_LINEAR)
        if self._segmentation_mask is None:
            self._segmentation_mask = mask
        else:
            self._segmentation_mask = (
                (1.0 - self.ema_alpha) * self._segmentation_mask + self.ema_alpha * mask
            )
        return self._segmentation_mask

    # ------------------------------------------------------------------
    # Math helpers
    # ------------------------------------------------------------------

    def _smooth_group(self, key: str, coords: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        prev = self._ema_store.get(key)
        smoothed: dict[str, np.ndarray] = {}
        for name, value in coords.items():
            val = value.astype(np.float32)
            if prev and name in prev:
                smoothed[name] = prev[name] * (1.0 - self.ema_alpha) + val * self.ema_alpha
            elif prev:
                smoothed[name] = prev[name] * (1.0 - self.ema_alpha) + val * self.ema_alpha
            else:
                smoothed[name] = val
        self._ema_store[key] = smoothed
        return smoothed

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a[:2] - b[:2]))

    def _compute_face_metrics(self, landmarks: dict[str, np.ndarray]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        try:
            le = landmarks["LEFT_EYE_OUTER"]
            re = landmarks["RIGHT_EYE_OUTER"]
            metrics["inter_pupil_dist"] = self._distance(le, re)
            metrics["head_roll"] = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))
        except KeyError:
            pass
        try:
            le = landmarks["LEFT_EAR"]
            re = landmarks["RIGHT_EAR"]
            metrics["face_width"] = self._distance(le, re)
        except KeyError:
            pass
        try:
            forehead = landmarks["FOREHEAD_CENTER"]
            chin = landmarks["CHIN"]
            metrics["face_height"] = self._distance(forehead, chin)
        except KeyError:
            pass
        if landmarks:
            center = np.mean([pt for pt in landmarks.values()], axis=0)
            metrics["center_x"] = float(center[0])
            metrics["center_y"] = float(center[1])
        return metrics

    def _compute_pose_metrics(self, landmarks: dict[str, np.ndarray]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        try:
            ls = landmarks["LEFT_SHOULDER"]
            rs = landmarks["RIGHT_SHOULDER"]
            metrics["shoulder_width"] = self._distance(ls, rs)
            metrics["shoulder_angle"] = math.degrees(math.atan2(rs[1] - ls[1], rs[0] - ls[0]))
        except KeyError:
            pass
        try:
            ls = landmarks["LEFT_SHOULDER"]
            rs = landmarks["RIGHT_SHOULDER"]
            lh = landmarks["LEFT_HIP"]
            rh = landmarks["RIGHT_HIP"]
            torso_center = np.mean([ls, rs, lh, rh], axis=0)
            metrics["torso_center_x"] = float(torso_center[0])
            metrics["torso_center_y"] = float(torso_center[1])
            metrics["torso_height"] = (self._distance(ls, lh) + self._distance(rs, rh)) / 2.0
        except KeyError:
            pass
        return metrics

    def _compute_hand_metrics(self, landmarks: dict[str, np.ndarray]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        try:
            thumb = landmarks["THUMB_TIP"]
            pinky = landmarks["PINKY_TIP"]
            metrics["span"] = self._distance(thumb, pinky)
        except KeyError:
            pass
        try:
            index = landmarks["INDEX_MCP"]
            pinky = landmarks["PINKY_MCP"]
            metrics["palm_width"] = self._distance(index, pinky)
        except KeyError:
            pass
        if "PALM_CENTER" in landmarks:
            metrics["palm_center_x"] = float(landmarks["PALM_CENTER"][0])
            metrics["palm_center_y"] = float(landmarks["PALM_CENTER"][1])
        return metrics

    def _detect_bowl_gesture(self, hand: dict[str, Any]) -> str | None:
        landmarks = hand.get("landmarks") or {}
        palm = landmarks.get("PALM_CENTER")
        if palm is None:
            return None
        palm_y = palm[1]
        fingers = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
        for finger in fingers:
            tip = landmarks.get(f"{finger}_TIP")
            if tip is None:
                return None
            if tip[1] >= palm_y:
                return None
        curled = True
        for finger in ["INDEX", "MIDDLE", "RING", "PINKY"]:
            tip = landmarks.get(f"{finger}_TIP")
            pip = landmarks.get(f"{finger}_PIP")
            if tip is None or pip is None:
                curled = False
                break
            if self._distance(tip, palm) > self._distance(pip, palm):
                curled = False
                break
        if not curled:
            return None
        thumb_tip = landmarks.get("THUMB_TIP")
        pinky_tip = landmarks.get("PINKY_TIP")
        pinky_mcp = landmarks.get("PINKY_MCP")
        index_mcp = landmarks.get("INDEX_MCP")
        if not all([thumb_tip is not None, pinky_tip is not None, pinky_mcp is not None, index_mcp is not None]):
            return None
        thumb_pinky = self._distance(thumb_tip, pinky_tip)
        palm_width = self._distance(index_mcp, pinky_mcp)
        if palm_width <= 1e-3:
            return None
        ratio = thumb_pinky / palm_width
        if not (0.6 <= ratio <= 1.2):
            return None
        label = hand.get("handedness", "Unknown")
        if "Left" in label:
            return "bowl_left"
        if "Right" in label:
            return "bowl_right"
        return "bowl_hand"


_ENGINE: AREngine | None = None
_CROWN_TRACKER: CrownTracker | None = None
_SPECIAL_EFFECTS = {"crown_basic"}


def get_engine() -> AREngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = AREngine()
    return _ENGINE


def _get_crown_tracker(crown_effect: Effect | None = None) -> CrownTracker | None:
    """Return a shared CrownTracker instance, initializing it on demand."""

    global _CROWN_TRACKER
    if _CROWN_TRACKER is not None:
        return _CROWN_TRACKER
    if mp is None:
        return None
    crown_image_path = None
    if crown_effect is not None:
        crown_image_path = str(crown_effect.sprite_path)
    try:
        _CROWN_TRACKER = CrownTracker(crown_image_path)
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        print("[DEBUG] CrownTracker init failed:", exc)
        _CROWN_TRACKER = None
    return _CROWN_TRACKER


# ---------------------------------------------------------------------------
# Effect definition / loader
# ---------------------------------------------------------------------------


CATEGORY_ORDER = ["頭", "顔", "コスメ", "衣装", "手", "背景", "パック"]
CURRENT_SEASON = "festival"
DEFAULT_PACKS: dict[str, list[str]] = {
    "学祭パック": [
        "crown_basic",
        "fairy_glow",
        "cheek_heart",
        "reitaku_emblem_hand",
        "bg_sakura",
    ]
}


class Effect:
    """AR effect defined by meta.json."""

    def __init__(self, meta_path: Path):
        with meta_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)
        self.meta_path = meta_path
        self.name: str = data["name"]
        self.display_name: str = data.get("display_name", self.name)
        self.description: str = data.get("description", "")
        self.category: str = data.get("category", "その他")
        self.priority: int = int(data.get("priority", 100))
        self.sprite_path = meta_path.parent / data.get("sprite", "sprite.png")
        ensure_sample_sprite(self.name, self.sprite_path)
        self.anchors: list[dict[str, Any]] = data.get("anchors", [])
        self.conditions: dict[str, Any] = data.get("conditions", {})
        self.animation: dict[str, Any] = data.get("animation", {})
        self.tags: list[str] = data.get("tags", [])
        self.sprite_frames: list[SpriteFrame] = self._load_sprite(self.sprite_path)

    # ------------------------------ sprite ------------------------------
    def _load_sprite(self, path: Path) -> list[SpriteFrame]:
        if not path.exists():
            raise FileNotFoundError(f"スプライトが見つかりません: {path}")
        image = Image.open(path)
        frames: list[SpriteFrame] = []
        if getattr(image, "is_animated", False):
            durations = image.info.get("duration", 100)
            if isinstance(durations, (int, float)):
                default_duration = max(float(durations) / 1000.0, 0.033)
            else:
                default_duration = 0.033
            for frame in ImageSequence.Iterator(image):
                duration_ms = frame.info.get("duration", default_duration * 1000)
                duration = max(float(duration_ms) / 1000.0, 0.033)
                frames.append(SpriteFrame(frame.convert("RGBA"), duration))
        else:
            frames.append(SpriteFrame(image.convert("RGBA"), 1 / 30))
        image.close()
        return frames

    def get_preview_image(self) -> Image.Image:
        return self.sprite_frames[0].image_rgba

    def frame_for_time(self, t: float) -> SpriteFrame:
        if len(self.sprite_frames) == 1:
            return self.sprite_frames[0]
        total = sum(frame.duration for frame in self.sprite_frames)
        if total <= 0:
            return self.sprite_frames[0]
        t_mod = t % total
        elapsed = 0.0
        for frame in self.sprite_frames:
            elapsed += frame.duration
            if t_mod <= elapsed:
                return frame
        return self.sprite_frames[-1]

    # --------------------------- condition check ------------------------
    def _season_ok(self) -> bool:
        seasons = self.conditions.get("season")
        if not seasons:
            return True
        if "all" in seasons:
            return True
        return CURRENT_SEASON in seasons

    def _gesture_ok(self, gestures: Iterable[str]) -> bool:
        required = self.conditions.get("gesture")
        if not required:
            return True
        if isinstance(required, str):
            required = [required]
        gestures_set = set(gestures)
        return any(req in gestures_set for req in required)

    def is_available(self, state: dict[str, Any]) -> bool:
        if not self._season_ok():
            return False
        if not self._gesture_ok(state.get("gestures", [])):
            return False
        return True

    # --------------------------- overlay render -------------------------
    def generate_overlays(self, state: dict[str, Any], t: float) -> list[Overlay]:
        if not state or not self.is_available(state):
            return []
        overlays: list[Overlay] = []
        sprite_frame = self.frame_for_time(t)
        for anchor_cfg in self.anchors:
            overlays.extend(self._generate_for_anchor(anchor_cfg, sprite_frame, state, t))
        return overlays

    def _generate_for_anchor(
        self, anchor_cfg: dict[str, Any], sprite_frame: SpriteFrame, state: dict[str, Any], t: float
    ) -> list[Overlay]:
        anchor_type = anchor_cfg.get("type")
        if anchor_type == "face":
            return self._generate_face_anchor(anchor_cfg, sprite_frame, state, t)
        if anchor_type == "pose":
            return self._generate_pose_anchor(anchor_cfg, sprite_frame, state, t)
        if anchor_type == "hand":
            return self._generate_hand_anchor(anchor_cfg, sprite_frame, state, t)
        if anchor_type == "background":
            return self._generate_background_anchor(anchor_cfg, sprite_frame, state)
        return []

    # --------------------------- helper contexts ------------------------
    @staticmethod
    def _distance_lookup(landmarks: dict[str, np.ndarray], a: str, b: str) -> float:
        pa = landmarks[a]
        pb = landmarks[b]
        return float(np.linalg.norm(pa[:2] - pb[:2]))

    @staticmethod
    def _angle_lookup(landmarks: dict[str, np.ndarray], a: str, b: str) -> float:
        pa = landmarks[a]
        pb = landmarks[b]
        return math.degrees(math.atan2(pb[1] - pa[1], pb[0] - pa[0]))

    def _build_context(self, base: dict[str, Any], extra: dict[str, float] | None = None) -> dict[str, Any]:
        context: dict[str, Any] = {}
        if extra:
            context.update(extra)
        context.update(base)
        context.update(
            {
                "math": math,
                "abs": abs,
                "min": min,
                "max": max,
            }
        )
        return context

    def _evaluate(self, expr: Any, context: dict[str, Any], default: float) -> float:
        if expr is None:
            return default
        if isinstance(expr, (int, float)):
            return float(expr)
        if isinstance(expr, str):
            try:
                return float(eval(expr, {"__builtins__": {}}, context))
            except Exception:
                return default
        return default

    def _prepare_sprite(
        self,
        sprite_frame: SpriteFrame,
        scale_factor: float,
        rotation_deg: float,
        horizontal_flip: bool,
    ) -> tuple[np.ndarray, int, int]:
        sprite = sprite_frame.image_rgba
        if horizontal_flip:
            sprite = sprite.transpose(Image.FLIP_LEFT_RIGHT)
        scale_factor = max(scale_factor, 1e-2)
        new_w = max(1, int(round(sprite.width * scale_factor)))
        new_h = max(1, int(round(sprite.height * scale_factor)))
        resized = sprite.resize((new_w, new_h), Image.LANCZOS)
        rotated = resized.rotate(rotation_deg, resample=Image.BICUBIC, expand=True)
        bgra = cv2.cvtColor(np.array(rotated), cv2.COLOR_RGBA2BGRA)
        return bgra, rotated.width, rotated.height

    def _animation_adjustments(self, overlay: Overlay, animation_cfg: dict[str, Any], t: float) -> None:
        if not animation_cfg:
            return
        speed = float(animation_cfg.get("speed_hz", 1.0))
        phase = float(animation_cfg.get("phase", 0.0))
        sine = math.sin(2 * math.pi * speed * t + phase)
        if "amplitude_px" in animation_cfg:
            amplitude = float(animation_cfg.get("amplitude_px", 0.0))
            overlay.y += int(round(amplitude * sine))
        if "alpha_pulse" in animation_cfg:
            min_a, max_a = animation_cfg.get("alpha_pulse", [0.5, 1.0])
            overlay.alpha_multiplier *= min_a + (max_a - min_a) * (sine * 0.5 + 0.5)

    # ------------------------- anchor implementations --------------------
    def _generate_face_anchor(
        self, anchor_cfg: dict[str, Any], sprite_frame: SpriteFrame, state: dict[str, Any], t: float
    ) -> list[Overlay]:
        face = state.get("face")
        if not face:
            return []
        landmarks: dict[str, np.ndarray] = face.get("landmarks", {})
        metrics: dict[str, float] = face.get("metrics", {})
        names = anchor_cfg.get("landmarks", [])
        points = [landmarks.get(name) for name in names if name in landmarks]
        if not points:
            return []
        base_context = {
            name: float(metrics.get(name, 0.0)) for name in metrics
        }
        base_context.update(
            {
                "distance": lambda a, b: self._distance_lookup(landmarks, a, b),
                "angle_deg": lambda a, b: self._angle_lookup(landmarks, a, b),
                "landmark_x": lambda name: float(landmarks[name][0]),
                "landmark_y": lambda name: float(landmarks[name][1]),
            }
        )
        context = self._build_context(base_context)
        transform = anchor_cfg.get("transform", {})
        sprite_width = sprite_frame.image_rgba.width
        scale_target = self._evaluate(transform.get("scale_from"), context, sprite_width)
        scale_factor = scale_target / max(sprite_width, 1)
        rotation = self._evaluate(transform.get("rotation_from"), context, metrics.get("head_roll", 0.0))
        offset_values = transform.get("offset_px", [0, 0])
        if isinstance(offset_values, list) and len(offset_values) == 2:
            offset_x = self._evaluate(offset_values[0], context, 0.0)
            offset_y = self._evaluate(offset_values[1], context, 0.0)
        else:
            offset_x = offset_y = 0.0
        horizontal_flip = bool(anchor_cfg.get("horizontal_flip", False))
        sprite_bgra, w, h = self._prepare_sprite(sprite_frame, scale_factor, rotation, horizontal_flip)
        center = np.mean(points, axis=0)
        rot_rad = math.radians(rotation)
        rot_matrix = np.array(
            [[math.cos(rot_rad), -math.sin(rot_rad)], [math.sin(rot_rad), math.cos(rot_rad)]]
        )
        offset_vec = rot_matrix @ np.array([offset_x, offset_y])
        center_xy = center[:2] + offset_vec
        top_left_x = int(round(center_xy[0] - w / 2))
        top_left_y = int(round(center_xy[1] - h / 2))
        overlay = Overlay(
            image=sprite_bgra,
            x=top_left_x,
            y=top_left_y,
            z_index=int(anchor_cfg.get("z_index", 0)),
            mask_with_segmentation=bool(anchor_cfg.get("mask_with_segmentation", False)),
            alpha_multiplier=float(anchor_cfg.get("opacity", 1.0)),
        )
        self._animation_adjustments(overlay, self.animation, t)
        return [overlay]

    def _generate_pose_anchor(
        self, anchor_cfg: dict[str, Any], sprite_frame: SpriteFrame, state: dict[str, Any], t: float
    ) -> list[Overlay]:
        pose = state.get("pose")
        if not pose:
            return []
        landmarks: dict[str, np.ndarray] = pose.get("landmarks", {})
        metrics: dict[str, float] = pose.get("metrics", {})
        names = anchor_cfg.get("landmarks", [])
        points = [landmarks.get(name) for name in names if name in landmarks]
        if not points:
            return []
        base_context = {name: float(value) for name, value in metrics.items()}
        base_context.update(
            {
                "distance": lambda a, b: self._distance_lookup(landmarks, a, b),
                "angle_deg": lambda a, b: self._angle_lookup(landmarks, a, b),
            }
        )
        context = self._build_context(base_context)
        transform = anchor_cfg.get("transform", {})
        sprite_width = sprite_frame.image_rgba.width
        scale_target = self._evaluate(transform.get("scale_from"), context, sprite_width)
        scale_factor = scale_target / max(sprite_width, 1)
        rotation = self._evaluate(transform.get("rotation_from"), context, metrics.get("shoulder_angle", 0.0))
        offset_values = transform.get("offset_px", [0, 0])
        if isinstance(offset_values, list) and len(offset_values) == 2:
            offset_x = self._evaluate(offset_values[0], context, 0.0)
            offset_y = self._evaluate(offset_values[1], context, 0.0)
        else:
            offset_x = offset_y = 0.0
        sprite_bgra, w, h = self._prepare_sprite(sprite_frame, scale_factor, rotation, False)
        center = np.mean(points, axis=0)
        rot_rad = math.radians(rotation)
        rot_matrix = np.array(
            [[math.cos(rot_rad), -math.sin(rot_rad)], [math.sin(rot_rad), math.cos(rot_rad)]]
        )
        offset_vec = rot_matrix @ np.array([offset_x, offset_y])
        center_xy = center[:2] + offset_vec
        top_left_x = int(round(center_xy[0] - w / 2))
        top_left_y = int(round(center_xy[1] - h / 2))
        overlay = Overlay(
            image=sprite_bgra,
            x=top_left_x,
            y=top_left_y,
            z_index=int(anchor_cfg.get("z_index", 0)),
            mask_with_segmentation=bool(anchor_cfg.get("mask_with_segmentation", False)),
            alpha_multiplier=float(anchor_cfg.get("opacity", 1.0)),
        )
        self._animation_adjustments(overlay, self.animation, t)
        return [overlay]

    def _generate_hand_anchor(
        self, anchor_cfg: dict[str, Any], sprite_frame: SpriteFrame, state: dict[str, Any], t: float
    ) -> list[Overlay]:
        hands = state.get("hands", [])
        if not hands:
            return []
        overlays: list[Overlay] = []
        requested_hands = anchor_cfg.get("handedness")
        transform = anchor_cfg.get("transform", {})
        for hand in hands:
            label = hand.get("handedness", "Unknown")
            if requested_hands and label not in requested_hands:
                continue
            landmarks: dict[str, np.ndarray] = hand.get("landmarks", {})
            metrics: dict[str, float] = hand.get("metrics", {})
            names = anchor_cfg.get("landmarks", [])
            points = [landmarks.get(name) for name in names if name in landmarks]
            if not points:
                continue
            base_context = {name: float(value) for name, value in metrics.items()}
            base_context.update(
                {
                    "distance": lambda a, b: self._distance_lookup(landmarks, a, b),
                    "angle_deg": lambda a, b: self._angle_lookup(landmarks, a, b),
                    "hand_rotation": float(hand.get("rotation", 0.0)),
                }
            )
            context = self._build_context(base_context)
            sprite_width = sprite_frame.image_rgba.width
            scale_target = self._evaluate(transform.get("scale_from"), context, sprite_width)
            scale_factor = scale_target / max(sprite_width, 1)
            rotation = self._evaluate(transform.get("rotation_from"), context, hand.get("rotation", 0.0))
            offset_values = transform.get("offset_px", [0, 0])
            if isinstance(offset_values, list) and len(offset_values) == 2:
                offset_x = self._evaluate(offset_values[0], context, 0.0)
                offset_y = self._evaluate(offset_values[1], context, 0.0)
            else:
                offset_x = offset_y = 0.0
            sprite_bgra, w, h = self._prepare_sprite(
                sprite_frame, scale_factor, rotation, bool(anchor_cfg.get("horizontal_flip", False))
            )
            center = np.mean(points, axis=0)
            rot_rad = math.radians(rotation)
            rot_matrix = np.array(
                [[math.cos(rot_rad), -math.sin(rot_rad)], [math.sin(rot_rad), math.cos(rot_rad)]]
            )
            offset_vec = rot_matrix @ np.array([offset_x, offset_y])
            center_xy = center[:2] + offset_vec
            top_left_x = int(round(center_xy[0] - w / 2))
            top_left_y = int(round(center_xy[1] - h / 2))
            overlay = Overlay(
                image=sprite_bgra,
                x=top_left_x,
                y=top_left_y,
                z_index=int(anchor_cfg.get("z_index", 0)),
                mask_with_segmentation=bool(anchor_cfg.get("mask_with_segmentation", False)),
                alpha_multiplier=float(anchor_cfg.get("opacity", 1.0)),
            )
            self._animation_adjustments(overlay, self.animation, t)
            overlays.append(overlay)
        return overlays

    def _generate_background_anchor(
        self, anchor_cfg: dict[str, Any], sprite_frame: SpriteFrame, state: dict[str, Any]
    ) -> list[Overlay]:
        frame_size = state.get("frame_size")
        if not frame_size:
            return []
        height, width = frame_size
        sprite = sprite_frame.image_rgba.resize((width, height), Image.LANCZOS)
        bgra = cv2.cvtColor(np.array(sprite), cv2.COLOR_RGBA2BGRA)
        overlay = Overlay(
            image=bgra,
            x=0,
            y=0,
            z_index=int(anchor_cfg.get("z_index", -100)),
            mask_with_segmentation=bool(anchor_cfg.get("mask_with_segmentation", True)),
            alpha_multiplier=float(anchor_cfg.get("opacity", 1.0)),
        )
        return [overlay]


_EFFECTS: dict[str, Effect] = {}
_EFFECTS_BY_CATEGORY: dict[str, list[Effect]] = {}


def load_effects(asset_root: str | Path | None = None) -> dict[str, Effect]:
    global _EFFECTS, _EFFECTS_BY_CATEGORY
    if _EFFECTS:
        return _EFFECTS
    if asset_root is None:
        asset_root = Path(__file__).parent / "assets" / "ar"
    root = Path(asset_root)
    if not root.exists():
        raise FileNotFoundError(f"ARアセットディレクトリが見つかりません: {root}")
    for meta_path in sorted(root.glob("*/meta.json")):
        effect = Effect(meta_path)
        _EFFECTS[effect.name] = effect
    _EFFECTS_BY_CATEGORY = {}
    for effect in _EFFECTS.values():
        _EFFECTS_BY_CATEGORY.setdefault(effect.category, []).append(effect)
    for effects in _EFFECTS_BY_CATEGORY.values():
        effects.sort(key=lambda e: (e.priority, e.display_name))
    return _EFFECTS


def get_effect(name: str) -> Effect | None:
    return _EFFECTS.get(name)


def get_effects_by_category() -> dict[str, list[Effect]]:
    categories = {key: list(effects) for key, effects in _EFFECTS_BY_CATEGORY.items()}
    for key in categories:
        categories[key].sort(key=lambda e: (e.priority, e.display_name))
    return categories


def resolve_effects(names: Sequence[str]) -> list[Effect]:
    return [effect for name in names if (effect := get_effect(name))]


def get_effect_packs() -> dict[str, list[str]]:
    return {name: list(effects) for name, effects in DEFAULT_PACKS.items()}


# ---------------------------------------------------------------------------
# Composition helper
# ---------------------------------------------------------------------------


def _compose_with_state(
    frame: np.ndarray,
    active_effects: Sequence[Effect],
    state: dict[str, Any] | None,
    timestamp: float,
    engine: AREngine,
) -> np.ndarray:
    if not active_effects or not state:
        return frame.copy()
    overlays: list[Overlay] = []
    for effect in active_effects:
        try:
            overlays.extend(effect.generate_overlays(state, timestamp))
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            engine._last_error = f"エフェクト '{effect.name}' の処理でエラー: {exc}"
    if not overlays:
        return frame.copy()
    overlays.sort(key=lambda ov: ov.z_index)
    composed = frame.copy()
    segmentation = state.get("segmentation")
    for overlay in overlays:
        composed = _blend_overlay(composed, overlay, segmentation)
    return composed


def apply(frame: np.ndarray, active_effects: Sequence[Effect], timestamp: float) -> np.ndarray:
    if not active_effects:
        # no AR overlays requested; don't touch Mediapipe at all
        print("[DEBUG] apply_ar_effects final effects:", [])
        return frame

    effect_names = [effect.name for effect in active_effects]
    residual_effects = [effect for effect in active_effects if effect.name not in _SPECIAL_EFFECTS]

    composed = frame.copy()
    engine: AREngine | None = None
    state: dict[str, Any] | None = None

    if residual_effects:
        engine = get_engine()
        engine.update_frame(frame, timestamp)
        state = engine.get_state()
        composed = _compose_with_state(composed, residual_effects, state, timestamp, engine)

    if "crown_basic" in effect_names:
        crown_effect = next((effect for effect in active_effects if effect.name == "crown_basic"), None)
        tracker = _get_crown_tracker(crown_effect)
        if tracker is not None:
            composed, _ = tracker.apply(composed, {"enable_crown": True})

    print("[DEBUG] apply_ar_effects final effects:", effect_names)
    return composed


def apply_to_still(
    frame: np.ndarray,
    active_effects: Sequence[Effect],
    timestamp: float | None = None,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Apply effects to a still image and return the composed frame and AR state."""

    if timestamp is None:
        timestamp = time.time()

    effect_names = [effect.name for effect in active_effects]
    residual_effects = [effect for effect in active_effects if effect.name not in _SPECIAL_EFFECTS]

    state: dict[str, Any] | None = None
    composed = frame.copy()

    if residual_effects:
        engine = get_engine()
        state = engine.process_static_frame(frame, timestamp)
        composed = _compose_with_state(composed, residual_effects, state, timestamp, engine)

    if "crown_basic" in effect_names:
        crown_effect = next((effect for effect in active_effects if effect.name == "crown_basic"), None)
        tracker = _get_crown_tracker(crown_effect)
        if tracker is not None:
            composed, _ = tracker.apply(composed, {"enable_crown": True})

    return composed, state


def extract_person_mask(
    frame: np.ndarray, threshold: float = 0.6
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Return a foreground mask (0-1 float) for the primary person in the frame."""

    engine = get_engine()
    state = engine.process_static_frame(frame, time.time())
    if not state:
        return None, None
    segmentation = state.get("segmentation")
    if segmentation is None:
        return None, state
    mask = np.clip(segmentation, 0.0, 1.0)
    if threshold is not None:
        mask = (mask >= threshold).astype(np.float32)
    else:
        mask = mask.astype(np.float32)
    return mask, state


def _blend_overlay(base: np.ndarray, overlay: Overlay, segmentation: np.ndarray | None) -> np.ndarray:
    h, w = base.shape[:2]
    oh, ow = overlay.image.shape[:2]
    x1 = max(0, overlay.x)
    y1 = max(0, overlay.y)
    x2 = min(w, overlay.x + ow)
    y2 = min(h, overlay.y + oh)
    if x1 >= x2 or y1 >= y2:
        return base
    region = base[y1:y2, x1:x2].astype(np.float32)
    overlay_crop = overlay.image[y1 - overlay.y : y2 - overlay.y, x1 - overlay.x : x2 - overlay.x]
    overlay_rgb = overlay_crop[..., :3].astype(np.float32)
    alpha = overlay_crop[..., 3].astype(np.float32) / 255.0
    alpha *= max(0.0, min(overlay.alpha_multiplier, 3.0))
    alpha = np.clip(alpha, 0.0, 1.0)
    if overlay.mask_with_segmentation and segmentation is not None:
        seg_crop = segmentation[y1:y2, x1:x2]
        if seg_crop.shape != alpha.shape:
            seg_crop = cv2.resize(seg_crop, (alpha.shape[1], alpha.shape[0]), interpolation=cv2.INTER_LINEAR)
        alpha *= 1.0 - np.clip(seg_crop, 0.0, 1.0)
    inv_alpha = 1.0 - alpha
    blended = overlay_rgb * alpha[..., None] + region * inv_alpha[..., None]
    base[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return base


__all__ = [
    "AREngine",
    "Effect",
    "apply",
    "apply_to_still",
    "extract_person_mask",
    "get_engine",
    "load_effects",
    "get_effect",
    "get_effects_by_category",
    "resolve_effects",
    "get_effect_packs",
]
