"""Privacy consent dialog shown before shooting.

This module implements a minimal consent gate that blocks the game
until players agree to the privacy notice required before shooting.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Callable, Optional

import customtkinter as ctk

CONSENT_TITLE = "撮影に進む前に（ご確認ください）"
CONSENT_BODY = """いつも遊んでくれてありがとう！\nこのゲームでは、安心して“ぷりくら体験”を楽しんでもらうため、次の情報を利用します。\n\n【収集する情報】\n・撮影した写真データ\n・ニックネーム（任意）\n・プレイ状況や端末情報（不具合改善のため）\n\n【利用目的】\n・撮影・加工・アルバム保存などのサービス提供\n・サービス改善のための分析（個人が特定されない形で利用します）\n\n【データの保存（クラウド保存）】\n・撮影した写真は、サービス提供のためクラウド上に保存されます。\n・データは安全に管理され、ご本人の許可なく外部に公開されることはありません。\n・学祭終了後、速やかにすべてのデータを削除します。\n\n【共有リンクについて（重要）】\n・写真をダウンロードするための専用リンクを発行できます。\n・そのリンクを他人に共有すると、他の人も写真を閲覧・保存できてしまいます。\n・リンクの有効期限は30分です。有効期限を過ぎるとアクセスできなくなります。\n\n【第三者提供】\n法律に基づく場合を除き、ご本人の同意なく外部へ提供することはありません。\n\n【未成年の方へ】\n13歳未満の方は、保護者の同意を得てご利用ください。\n\nくわしくは「プライバシーポリシー」をご確認ください。"""

_SESSION_STATE_PATH = Path(tempfile.gettempdir()) / "purikura_privacy_consent_state.json"


class PrivacyConsentDialog(ctk.CTkToplevel):
    """Modal dialog requesting privacy consent before shooting."""

    def __init__(self, parent: ctk.CTk, on_agree: Callable[[], None], on_decline: Callable[[], None]):
        super().__init__(parent)
        self.title(CONSENT_TITLE)
        self.transient(parent)
        self.resizable(False, False)
        self._on_agree = on_agree
        self._on_decline = on_decline

        self.protocol("WM_DELETE_WINDOW", lambda: None)
        self.bind("<Escape>", lambda event: "break")

        container = ctk.CTkFrame(self, corner_radius=12)
        container.pack(fill="both", expand=True, padx=24, pady=24)

        header = ctk.CTkLabel(container, text=CONSENT_TITLE, font=ctk.CTkFont(size=20, weight="bold"))
        header.pack(pady=(8, 12))

        body_frame = ctk.CTkScrollableFrame(container, width=520, height=360, corner_radius=10)
        body_frame.pack(fill="both", expand=True, pady=(0, 16))
        body_label = ctk.CTkLabel(
            body_frame,
            text=CONSENT_BODY,
            justify="left",
            anchor="w",
            wraplength=500,
        )
        body_label.pack(fill="both", expand=True, padx=6, pady=6)

        button_row = ctk.CTkFrame(container)
        button_row.pack(pady=(0, 4))
        self.agree_button = ctk.CTkButton(
            button_row,
            text="同意して撮影へ進む",
            width=220,
            command=self._handle_agree,
        )
        self.agree_button.grid(row=0, column=0, padx=6, pady=6)
        decline_button = ctk.CTkButton(
            button_row,
            text="同意しない（タイトルに戻る）",
            width=220,
            command=self._handle_decline,
        )
        decline_button.grid(row=0, column=1, padx=6, pady=6)

        self.columnconfigure(0, weight=1)
        button_row.columnconfigure(0, weight=1)
        button_row.columnconfigure(1, weight=1)

        self.grab_set()
        self.after(50, self._focus_primary_button)
        self.lift()
        parent.update_idletasks()
        self._center_over_parent(parent)

        self.bind("<Return>", lambda event: self._handle_agree())
        self.bind("<space>", lambda event: self._handle_agree())

    def _center_over_parent(self, parent: ctk.CTk) -> None:
        width = 600
        height = 520
        try:
            parent_width = parent.winfo_width()
            parent_height = parent.winfo_height()
            parent_x = parent.winfo_rootx()
            parent_y = parent.winfo_rooty()
        except Exception:
            parent_width = parent.winfo_screenwidth()
            parent_height = parent.winfo_screenheight()
            parent_x = 0
            parent_y = 0
        if parent_width <= 1 or parent_height <= 1:
            parent_width = parent.winfo_screenwidth()
            parent_height = parent.winfo_screenheight()
            parent_x = 0
            parent_y = 0
        x = parent_x + max((parent_width - width) // 2, 0)
        y = parent_y + max((parent_height - height) // 2, 0)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _focus_primary_button(self) -> None:
        try:
            self.agree_button.focus_set()
        except Exception:
            pass

    def _handle_agree(self) -> None:
        self.grab_release()
        self.destroy()
        self._on_agree()

    def _handle_decline(self) -> None:
        self.grab_release()
        self.destroy()
        self._on_decline()


class DeclineOverlay(ctk.CTkToplevel):
    """Full-screen overlay shown when the user declines the consent."""

    def __init__(self, parent: ctk.CTk):
        super().__init__(parent)
        self.overrideredirect(True)
        try:
            self.attributes("-topmost", True)
        except Exception:
            pass
        self.configure(fg_color=("#101010", "#101010"))
        self.bind("<Escape>", lambda event: "break")
        self.bind("<Return>", lambda event: "break")
        self.bind("<space>", lambda event: "break")

        parent.update_idletasks()
        width = parent.winfo_width()
        height = parent.winfo_height()
        x = parent.winfo_rootx()
        y = parent.winfo_rooty()
        if width <= 1 or height <= 1:
            width = parent.winfo_screenwidth()
            height = parent.winfo_screenheight()
            x = 0
            y = 0
        self.geometry(f"{width}x{height}+{x}+{y}")

        message = ctk.CTkLabel(
            self,
            text="また遊んでね",
            font=ctk.CTkFont(size=32, weight="bold"),
            anchor="center",
        )
        message.pack(expand=True, fill="both", padx=24, pady=24)

        self.grab_set()
        self.focus_force()


class PrivacyConsentManager:
    """Coordinator that ensures privacy consent before shooting."""

    def __init__(self, root: ctk.CTk):
        self.root = root
        self._dialog: Optional[PrivacyConsentDialog] = None
        self._overlay: Optional[DeclineOverlay] = None
        self._pending_callback: Optional[Callable[[], None]] = None
        self._agreed = False
        self._declined = False
        self._load_state()

    def _load_state(self) -> None:
        if not _SESSION_STATE_PATH.exists():
            return
        try:
            data = json.loads(_SESSION_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            try:
                _SESSION_STATE_PATH.unlink()
            except Exception:
                pass
            return
        if data.get("pid") == os.getpid() and data.get("agreed"):
            self._agreed = True
        else:
            try:
                _SESSION_STATE_PATH.unlink()
            except Exception:
                pass

    def _save_state(self) -> None:
        if not self._agreed:
            return
        payload = {"pid": os.getpid(), "agreed": True}
        try:
            _SESSION_STATE_PATH.write_text(json.dumps(payload), encoding="utf-8")
        except Exception:
            pass

    @property
    def is_declined(self) -> bool:
        return self._declined

    def ensure_consent(self, on_accept: Optional[Callable[[], None]] = None) -> bool:
        if self._declined:
            return False
        if self._agreed:
            return True
        if on_accept is not None:
            self._pending_callback = on_accept
        if self._dialog is None or not self._dialog.winfo_exists():
            self._dialog = PrivacyConsentDialog(self.root, self._handle_agree, self._handle_decline)
        else:
            self._dialog.lift()
            self._dialog.focus_force()
        return False

    def _handle_agree(self) -> None:
        self._agreed = True
        self._save_state()
        callback = self._pending_callback
        self._pending_callback = None
        if callback:
            self.root.after(0, callback)

    def _handle_decline(self) -> None:
        self._declined = True
        try:
            _SESSION_STATE_PATH.unlink()
        except Exception:
            pass
        if self._overlay is None or not self._overlay.winfo_exists():
            self._overlay = DeclineOverlay(self.root)
        else:
            self._overlay.lift()
            self._overlay.focus_force()
