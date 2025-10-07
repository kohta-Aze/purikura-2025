"""draw.coords
=================
座標変換およびデバイスピクセル比補正のユーティリティ。

編集画面では表示画像と元画像のサイズが異なるため、
座標の正規化・逆変換を行う補助関数をまとめる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import customtkinter as ctk


@dataclass
class CoordinateContext:
    """座標変換に必要なパラメータをまとめたデータクラス。

    Attributes
    ----------
    image_size : tuple[int, int]
        元画像の幅・高さ
    display_size : tuple[int, int]
        画面上に表示されているCTkImageの幅・高さ
    dpr : float
        デバイスピクセル比（tk scaling）。
    """

    image_size: Tuple[int, int]
    display_size: Tuple[int, int]
    dpr: float = 1.0


def get_device_pixel_ratio(widget: ctk.CTkBaseClass) -> float:
    """ウィジェットからtk scalingを取得し、DPRを返す。

    param widget: 計測対象のTkウィジェット
    output: デバイスピクセル比（float）
    """

    try:
        scaling = float(widget.winfo_toplevel().tk.call("tk", "scaling"))
    except Exception:
        scaling = 1.0
    return scaling if scaling > 0 else 1.0


def display_to_image_coords(x: float, y: float, context: CoordinateContext) -> tuple[float, float]:
    """表示座標を元画像座標へ変換する。

    param x: 表示領域でのX座標
    param y: 表示領域でのY座標
    param context: 座標変換コンテキスト
    output: (x, y) in 元画像座標
    """

    disp_w, disp_h = context.display_size
    img_w, img_h = context.image_size
    if disp_w == 0 or disp_h == 0:
        return 0.0, 0.0
    scale_x = img_w / (disp_w * context.dpr)
    scale_y = img_h / (disp_h * context.dpr)
    return x * scale_x, y * scale_y


def normalize_coords(x: float, y: float, context: CoordinateContext) -> tuple[float, float]:
    """元画像座標を0〜1の正規化座標へ変換する。

    param x: 元画像基準のX座標
    param y: 元画像基準のY座標
    param context: 座標変換コンテキスト
    output: (u, v) ∈ [0,1]
    """

    img_w, img_h = context.image_size
    if img_w == 0 or img_h == 0:
        return 0.0, 0.0
    return x / img_w, y / img_h


def denormalize_coords(u: float, v: float, context: CoordinateContext) -> tuple[float, float]:
    """正規化座標を元画像座標へ戻す。

    param u: 正規化X
    param v: 正規化Y
    param context: 座標変換コンテキスト
    output: (x, y) in 元画像座標
    """

    img_w, img_h = context.image_size
    return u * img_w, v * img_h
