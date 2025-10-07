"""ar.crown_tracker
====================
MediaPipe FaceMesh を用いた王冠オーバーレイ生成。

顔ランドマークから頭頂位置・スケール・回転角を推定し、
EMAで平滑化したのち王冠PNGを合成する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - 実行環境に依存
    mp = None


@dataclass
class CrownState:
    """1顔ごとの平滑化済みパラメータを保持するデータ。"""

    center: Tuple[float, float]
    anchor: Tuple[float, float]
    scale: float
    angle: float


class CrownTracker:
    """王冠オーバーレイを生成するトラッカー。"""

    def __init__(self, crown_image_path: str | None = None, smoothing_alpha: float = 0.2):
        """コンストラクタ。

        param crown_image_path: 王冠PNGのパス（未指定時は簡易王冠を生成）
        param smoothing_alpha: EMA平滑化係数
        """

        self.smoothing_alpha = smoothing_alpha
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) if mp else None
        self.crown_image = self._load_crown_image(crown_image_path)
        self._previous_states: list[CrownState] = []

    def reset(self):
        """平滑化状態を破棄する。"""

        self._previous_states = []

    def apply(self, frame_bgr: np.ndarray, settings: dict[str, object]) -> tuple[np.ndarray, list[dict[str, float]]]:
        """王冠を合成しつつ、顔中心情報を返す。

        param frame_bgr: OpenCVのBGRフレーム
        param settings: {'enable_crown': bool, 'ema_alpha': float}
        output: (合成済みフレーム, [{'center': (x,y), 'radius': r}...])
        """

        if self.face_mesh is None:
            return frame_bgr.copy(), []
        enable = bool(settings.get('enable_crown', True))
        alpha = float(settings.get('ema_alpha', self.smoothing_alpha))
        if not enable:
            self.reset()
            return frame_bgr.copy(), []

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            self.reset()
            return frame_bgr.copy(), []

        output = frame_bgr.copy()
        h, w, _ = output.shape
        face_centers: list[dict[str, float]] = []
        new_states: list[CrownState] = []

        for idx, landmarks in enumerate(results.multi_face_landmarks):
            params = self._estimate_params(landmarks.landmark, w, h)
            if params is None:
                continue
            prev = self._previous_states[idx] if idx < len(self._previous_states) else None
            smoothed = self._smooth_params(prev, params, alpha)
            new_states.append(smoothed)
            face_centers.append({
                'center_x': smoothed.center[0],
                'center_y': smoothed.center[1],
                'radius': params['radius'],
            })
            output = self._overlay_crown(output, smoothed, params['radius'])

        self._previous_states = new_states
        return output, face_centers

    # ------------------------------------------------------------------
    def _load_crown_image(self, crown_image_path: str | None) -> Image.Image:
        """王冠PNGを読み込む。無ければ簡易王冠を自作する。"""

        if crown_image_path and Path(crown_image_path).exists():
            return Image.open(crown_image_path).convert("RGBA")
        # デフォルトの簡易王冠生成
        img = Image.new("RGBA", (320, 200), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        base_color = (255, 215, 0, 255)
        points = [(10, 180), (70, 60), (130, 180), (190, 60), (250, 180), (310, 60), (310, 200), (10, 200)]
        draw.polygon(points, fill=base_color)
        for x in (70, 190, 310):
            draw.ellipse((x - 20, 40, x + 20, 80), fill=(255, 105, 180, 255))
        draw.rectangle((10, 170, 310, 200), fill=(255, 200, 0, 255))
        return img

    def _estimate_params(self, landmarks, width: int, height: int):
        """FaceMeshランドマークから幾何パラメータを推定する。"""

        indices = {
            'left': 127,
            'right': 356,
            'top': 10,
            'chin': 152,
            'nose': 1,
        }
        try:
            left = landmarks[indices['left']]
            right = landmarks[indices['right']]
            top = landmarks[indices['top']]
            chin = landmarks[indices['chin']]
        except IndexError:
            return None

        center_x = (left.x + right.x) / 2 * width
        top_y = top.y * height
        chin_y = chin.y * height
        face_height = max(chin_y - top_y, 1.0)
        radius = max(abs(right.x - left.x) * width / 2, face_height / 4)
        anchor_y = top_y - face_height * 0.25
        angle = math.degrees(math.atan2((right.y - left.y), (right.x - left.x)))
        scale = radius * 2 / max(self.crown_image.width, 1)
        return {
            'center': (center_x, anchor_y),
            'anchor': (center_x, top_y),
            'scale': scale,
            'angle': angle,
            'radius': radius,
        }

    def _smooth_params(self, previous: CrownState | None, params: dict, alpha: float) -> CrownState:
        """EMAで中心・スケール・角度を平滑化する。"""

        if previous is None:
            return CrownState(
                center=params['center'],
                anchor=params['anchor'],
                scale=params['scale'],
                angle=params['angle'],
            )
        def ema(old, new):
            return (1 - alpha) * old + alpha * new
        center = (ema(previous.center[0], params['center'][0]), ema(previous.center[1], params['center'][1]))
        anchor = (ema(previous.anchor[0], params['anchor'][0]), ema(previous.anchor[1], params['anchor'][1]))
        scale = ema(previous.scale, params['scale'])
        angle = ema(previous.angle, params['angle'])
        return CrownState(center=center, anchor=anchor, scale=scale, angle=angle)

    def _overlay_crown(self, frame_bgr: np.ndarray, state: CrownState, radius: float) -> np.ndarray:
        """平滑化されたパラメータをもとに王冠PNGを合成する。"""

        crown_rgba = cv2.cvtColor(np.array(self.crown_image), cv2.COLOR_RGBA2BGRA)
        scale = max(state.scale, 0.2)
        target_w = max(int(crown_rgba.shape[1] * scale), 1)
        target_h = max(int(crown_rgba.shape[0] * scale), 1)
        resized = cv2.resize(crown_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # 回転行列を作成（中心回り）
        center = (target_w / 2, target_h / 2)
        M = cv2.getRotationMatrix2D(center, state.angle, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        nW = int((target_h * sin) + (target_w * cos))
        nH = int((target_h * cos) + (target_w * sin))
        M[0, 2] += (nW / 2) - center[0]
        M[1, 2] += (nH / 2) - center[1]
        rotated = cv2.warpAffine(resized, M, (nW, nH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

        # 合成位置（anchorは頭頂付近）
        anchor_x, anchor_y = state.anchor
        top_left_x = int(anchor_x - nW / 2)
        top_left_y = int(anchor_y - nH * 0.9)
        return self._alpha_blend(frame_bgr, rotated, top_left_x, top_left_y)

    def _alpha_blend(self, base: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:
        """BGRAオーバーレイを位置指定で合成する。"""

        h, w = overlay.shape[:2]
        frame_h, frame_w = base.shape[:2]
        if x >= frame_w or y >= frame_h or x + w <= 0 or y + h <= 0:
            return base
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + w, frame_w)
        y2 = min(y + h, frame_h)
        overlay_x1 = x1 - x
        overlay_y1 = y1 - y
        overlay_x2 = overlay_x1 + (x2 - x1)
        overlay_y2 = overlay_y1 + (y2 - y1)

        roi = base[y1:y2, x1:x2]
        overlay_roi = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
        alpha = overlay_roi[..., 3:4] / 255.0
        blended = (1.0 - alpha) * roi + alpha * overlay_roi[..., :3]
        base[y1:y2, x1:x2] = blended.astype(np.uint8)
        return base
