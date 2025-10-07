"""draw.smoothing
==================
描画ストローク用の補間・予測ユーティリティ。

短いバッファに蓄積した座標列から、線形/Catmull-Rom補間や
EMA/EWMA/Kalmanによる予測を行い、なめらかなペン描画を実現する。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Sequence, Tuple

import numpy as np

PointT = Tuple[float, float, float]


def exponential_moving_average(previous: float | None, current: float, alpha: float) -> float:
    """指数移動平均（EMA）を1ステップ計算する。

    param previous: 直前のEMA値。Noneならcurrentを返す
    param current: 今回の観測値
    param alpha: 平滑化係数（0 < α ≤ 1）
    output: EMA結果
    """

    if previous is None:
        return current
    return (1.0 - alpha) * previous + alpha * current


def catmull_rom_spline(points: Sequence[PointT], resolution: int = 12) -> List[Tuple[float, float]]:
    """Catmull-Romスプラインで区間を補間する。

    param points: (x, y, t) × 4 の制御点
    param resolution: 0〜1を何分割するか
    output: 補間後の座標列（x, y）
    """

    if len(points) < 4:
        raise ValueError("Catmull-Rom補間には4点が必要です")
    p0, p1, p2, p3 = points[-4:]
    result: List[Tuple[float, float]] = []
    for i in range(resolution + 1):
        t = i / resolution
        x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t * t + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t * t * t)
        y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t * t + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t * t * t)
        result.append((x, y))
    return result


def linear_segment(points: Sequence[PointT]) -> List[Tuple[float, float]]:
    """線形補間で2点間のセグメントを生成する。

    param points: 2点の制御点
    output: 補間済み座標列（x, y）
    """

    if len(points) < 2:
        return []
    (x0, y0, _), (x1, y1, _) = points[-2:]
    steps = max(int(math.hypot(x1 - x0, y1 - y0) // 2), 1)
    return [
        (x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps)
        for i in range(1, steps + 1)
    ]


@dataclass
class KalmanFilter2D:
    """一定速度モデルの2次元カルマンフィルタ。"""

    process_noise: float = 1e-3
    measurement_noise: float = 0.05

    def __post_init__(self):
        self.x = np.zeros((4, 1), dtype=float)
        self.P = np.eye(4, dtype=float)
        self.initialized = False

    def reset(self, x: float, y: float):
        """状態ベクトルを初期化する。"""

        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=float)
        self.P = np.eye(4, dtype=float)
        self.initialized = True

    def predict(self, dt: float) -> Tuple[float, float]:
        """指定dt先を予測し、予測座標を返す。"""

        if not self.initialized:
            return 0.0, 0.0
        F = np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=float,
        )
        Q = self.process_noise * np.eye(4, dtype=float)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, x: float, y: float):
        """観測値で状態を更新する。"""

        if not self.initialized:
            self.reset(x, y)
            return
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        R = self.measurement_noise * np.eye(2, dtype=float)
        z = np.array([[x], [y]], dtype=float)
        y_residual = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y_residual
        I = np.eye(4, dtype=float)
        self.P = (I - K @ H) @ self.P


class StrokeSmoother:
    """描画ストロークを蓄積し、補間済み座標列を生成するマネージャ。"""

    def __init__(
        self,
        method: str = "catmull",
        prediction_mode: str = "off",
        ema_alpha: float = 0.2,
        buffer_size: int = 16,
        resolution: int = 12,
    ):
        """コンストラクタ。

        param method: "catmull" or "linear"
        param prediction_mode: "off"/"ewma"/"kalman"
        param ema_alpha: EMA係数
        param buffer_size: 保持するサンプル数
        param resolution: Catmull-Rom用の分割数
        """

        self.method = method
        self.prediction_mode = prediction_mode
        self.ema_alpha = ema_alpha
        self.resolution = resolution
        self.buffer: Deque[PointT] = deque(maxlen=buffer_size)
        self.last_point: Tuple[float, float] | None = None
        self.ema_velocity = (0.0, 0.0)
        self.kalman = KalmanFilter2D()
        self.last_timestamp = None

    def reset(self):
        """内部状態をリセットする。"""

        self.buffer.clear()
        self.last_point = None
        self.ema_velocity = (0.0, 0.0)
        self.kalman = KalmanFilter2D()
        self.last_timestamp = None

    def configure(self, method: str | None = None, prediction_mode: str | None = None, ema_alpha: float | None = None):
        """補間モードやパラメータを動的に設定する。"""

        if method:
            self.method = method
        if prediction_mode:
            self.prediction_mode = prediction_mode
        if ema_alpha is not None:
            self.ema_alpha = ema_alpha

    def add_sample(self, point: PointT) -> List[Tuple[float, float]]:
        """サンプルを追加し、新しく描画すべき座標列を返す。

        param point: (x, y, timestamp)
        output: 描画対象の座標列
        """

        self.buffer.append(point)
        new_points: List[Tuple[float, float]] = []
        predicted = None
        if len(self.buffer) >= 2:
            predicted = self._predict_next(point)
        if self.method == "catmull" and len(self.buffer) >= 4:
            ctrl = list(self.buffer)[-4:]
            if predicted:
                ctrl = ctrl[:-1] + [(predicted[0], predicted[1], ctrl[-1][2])]
            new_points = catmull_rom_spline(ctrl, resolution=self.resolution)
        elif len(self.buffer) >= 2:
            ctrl = list(self.buffer)[-2:]
            if predicted:
                ctrl.append((predicted[0], predicted[1], ctrl[-1][2]))
            new_points = linear_segment(ctrl)

        filtered: List[Tuple[float, float]] = []
        for pt in new_points:
            if self.last_point is None or math.hypot(pt[0] - self.last_point[0], pt[1] - self.last_point[1]) >= 0.5:
                filtered.append(pt)
                self.last_point = pt
        return filtered

    def _predict_next(self, point: PointT) -> Tuple[float, float] | None:
        """予測モードに応じて次位置を推定する。"""

        if self.prediction_mode == "off":
            return None
        if len(self.buffer) < 2:
            return None
        x, y, t = point
        prev_x, prev_y, prev_t = self.buffer[-2]
        dt = max(t - prev_t, 1e-3)
        vx = (x - prev_x) / dt
        vy = (y - prev_y) / dt
        if self.prediction_mode == "ewma":
            ema_vx = exponential_moving_average(self.ema_velocity[0], vx, self.ema_alpha)
            ema_vy = exponential_moving_average(self.ema_velocity[1], vy, self.ema_alpha)
            self.ema_velocity = (ema_vx, ema_vy)
            horizon = dt
            return x + ema_vx * horizon, y + ema_vy * horizon
        if self.prediction_mode == "kalman":
            if not self.kalman.initialized:
                self.kalman.reset(x, y)
            else:
                self.kalman.predict(dt)
            self.kalman.update(x, y)
            predicted_x, predicted_y = self.kalman.predict(dt)
            return predicted_x, predicted_y
        return None
