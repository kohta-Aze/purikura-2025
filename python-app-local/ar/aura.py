"""ar.aura
==========
被写体周囲に波動（オーラ）エフェクトを描画するモジュール。

王冠トラッカーから得られた顔位置をもとに、リングやパーティクルを
描画して視覚効果を追加する。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np


@dataclass
class AuraSettings:
    """波動エフェクトのパラメータ設定。"""

    color: tuple[int, int, int] = (255, 120, 220)
    intensity: float = 0.6
    frequency: float = 1.5
    spread_speed: float = 2.0
    mix_mode: str = "rgb_add"


class AuraRenderer:
    """円環/パーティクルの描画を担当するレンダラー。"""

    def __init__(self):
        self.frame_index = 0

    def apply(self, frame_bgr: np.ndarray, faces: List[dict], settings: dict[str, object]) -> np.ndarray:
        """フレームへ波動エフェクトを描画する。

        param frame_bgr: OpenCVのBGR画像
        param faces: CrownTrackerから得た中心情報の配列
        param settings: AuraSettingsに相当する辞書
        output: エフェクト合成後のフレーム
        """

        enable = bool(settings.get('enable_aura', True))
        if not enable or not faces:
            return frame_bgr

        aura_settings = AuraSettings(
            color=tuple(settings.get('aura_color', (255, 120, 220))),
            intensity=float(settings.get('aura_intensity', 0.6)),
            frequency=float(settings.get('aura_frequency', 1.5)),
            spread_speed=float(settings.get('aura_spread', 2.0)),
            mix_mode=str(settings.get('aura_mix_mode', 'rgb_add')),
        )
        overlays = []
        alpha_maps = []
        for face in faces:
            overlay, alpha_map = self._render_single(frame_bgr.shape, face, aura_settings)
            overlays.append(overlay)
            alpha_maps.append(alpha_map)

        if not overlays:
            return frame_bgr

        combined_rgb, combined_alpha = self._combine(overlays, alpha_maps, aura_settings.mix_mode)
        result = self._blend(frame_bgr, combined_rgb, combined_alpha)
        self.frame_index += 1
        return result

    # ------------------------------------------------------------------
    def _render_single(self, frame_shape, face: dict, settings: AuraSettings):
        """1被写体分のオーラを生成する。"""

        h, w, _ = frame_shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        alpha = np.zeros((h, w), dtype=np.uint8)
        center = (int(face['center_x']), int(face['center_y']))
        base_radius = max(int(face['radius'] * 1.1), 40)
        phase = self.frame_index * settings.frequency
        max_rings = 3
        for i in range(max_rings):
            ring_phase = phase + i * 0.7
            radius = int(base_radius + (i * settings.spread_speed * 12) + math.sin(ring_phase) * base_radius * 0.2)
            thickness = max(int(base_radius * 0.08), 2)
            ring_alpha = int(min(255, 120 + settings.intensity * 80 - i * 20))
            cv2.circle(overlay, center, radius, settings.color, thickness=thickness, lineType=cv2.LINE_AA)
            cv2.circle(alpha, center, radius, ring_alpha, thickness=thickness, lineType=cv2.LINE_AA)
            for theta in np.linspace(0, 2 * math.pi, 18, endpoint=False):
                particle_radius = radius + int(settings.spread_speed * 6)
                px = int(center[0] + particle_radius * math.cos(theta + ring_phase / 4))
                py = int(center[1] + particle_radius * math.sin(theta + ring_phase / 4))
                cv2.circle(overlay, (px, py), 3, settings.color, thickness=-1)
                cv2.circle(alpha, (px, py), 3, int(ring_alpha * 0.8), thickness=-1)
        return overlay, alpha

    def _combine(self, overlays: List[np.ndarray], alphas: List[np.ndarray], mode: str):
        """複数のオーバーレイを合成する。"""

        if not overlays:
            return np.zeros_like(overlays[0]), np.zeros_like(alphas[0])
        rgb = overlays[0].copy()
        alpha = alphas[0].astype(np.float32)
        if mode == 'rgb_add':
            for ov, a in zip(overlays[1:], alphas[1:]):
                rgb = cv2.add(rgb, ov)
                alpha = np.clip(alpha + a, 0, 255)
            return rgb, alpha.astype(np.uint8)

        # HSV補間
        hsv_weight = None
        hsv_sum = None
        value_max = None
        for ov, a in zip(overlays, alphas):
            hsv = cv2.cvtColor(ov, cv2.COLOR_BGR2HSV).astype(np.float32)
            weight = (a.astype(np.float32) / 255.0)
            if hsv_sum is None:
                hsv_sum = hsv * weight[..., None]
                hsv_weight = weight
                value_max = hsv[..., 2]
            else:
                hsv_sum += hsv * weight[..., None]
                hsv_weight += weight
                value_max = np.maximum(value_max, hsv[..., 2])
        hsv_weight = np.clip(hsv_weight, 1e-6, None)
        hsv_mean = np.zeros_like(hsv_sum)
        hsv_mean[..., 0] = hsv_sum[..., 0] / hsv_weight
        hsv_mean[..., 1] = hsv_sum[..., 1] / hsv_weight
        hsv_mean[..., 2] = value_max
        rgb = cv2.cvtColor(hsv_mean.astype(np.uint8), cv2.COLOR_HSV2BGR)
        alpha = np.clip(hsv_weight * 255.0, 0, 255).astype(np.uint8)
        return rgb, alpha

    def _blend(self, base: np.ndarray, overlay: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        """アルファブレンドでフレームへ重ねる。"""

        alpha_f = alpha.astype(np.float32) / 255.0
        alpha_f = alpha_f[..., None]
        blended = base.astype(np.float32) * (1.0 - alpha_f) + overlay.astype(np.float32) * alpha_f
        np.clip(blended, 0, 255, out=blended)
        return blended.astype(np.uint8)
