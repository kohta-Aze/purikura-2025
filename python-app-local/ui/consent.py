"""ui.consent
=============
共有許諾を取得するモーダルダイアログ。
"""

from __future__ import annotations

from typing import Callable, Optional

from PIL import Image
import customtkinter as ctk


class ShareConsentDialog(ctk.CTkToplevel):
    """共有許諾を問い合わせるモーダル。"""

    def __init__(self, parent, on_decision: Callable[[bool], None], preview_image: Optional[Image.Image] = None):
        """コンストラクタ。

        param parent: 親ウィンドウ
        param on_decision: True/Falseを受け取るコールバック
        param preview_image: 任意のプレビューPIL画像
        """

        super().__init__(parent)
        self.title("作例として共有してもよいですか？")
        self.geometry("520x420")
        self.resizable(False, False)
        self.on_decision = on_decision

        ctk.CTkLabel(self, text="この写真をアプリの作例として掲示してもよいですか？", wraplength=460,
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(24, 12))

        if preview_image is not None:
            preview = preview_image.copy()
            preview.thumbnail((400, 240))
            ctk_img = ctk.CTkImage(light_image=preview, dark_image=preview, size=preview.size)
            ctk.CTkLabel(self, image=ctk_img, text="").pack(pady=(0, 16))
            self.preview_ref = ctk_img
        else:
            ctk.CTkLabel(self, text="YES を選ぶと samples/ に縮小画像と編集メタが保存されます。",
                         wraplength=420).pack(pady=(0, 16))

        btn_row = ctk.CTkFrame(self)
        btn_row.pack(pady=(8, 24))
        yes_btn = ctk.CTkButton(btn_row, text="YES", width=140, command=self._on_yes)
        yes_btn.grid(row=0, column=0, padx=12)
        no_btn = ctk.CTkButton(btn_row, text="NO", width=140, command=self._on_no)
        no_btn.grid(row=0, column=1, padx=12)

        self.grab_set()
        self.focus_force()

    def _on_yes(self):
        """YESが選ばれたときの処理。"""

        self.on_decision(True)
        self.destroy()

    def _on_no(self):
        """NOが選ばれたときの処理。"""

        self.on_decision(False)
        self.destroy()


def request_share_consent(parent, callback: Callable[[bool], None], preview_image: Optional[Image.Image] = None):
    """共有許諾モーダルを開き、結果をコールバックする。"""

    ShareConsentDialog(parent, callback, preview_image)
