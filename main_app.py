# main_app.py (UIをcustomtkinterでモダン化・機能/設計/AWS連携は維持)
import io
import queue
import threading
import time
import json
import cv2
import numpy as np
import qrcode
import requests
from PIL import Image, ImageTk

import customtkinter as ctk

# 既存の専門家たち（責務・インターフェースは維持）
from editor_app import (
    AssetManager, CameraCapture, ImageCompositor, ImageEditor, ImageSaver,
    EditingFrame, FinalSelectionFrame, PhotoSelectionFrame, PostEditChoiceFrame,
    ShootingFrame, StartFrame, ThankYouFrame
)

# =========================================================
# 変更禁止：AWSアップロード仕様はそのまま（メソッド名/挙動も同一）
# =========================================================
class ImageUploader:
    """AWSに画像をアップロードする専門家（API Gateway + presigned URL）"""
    def __init__(self, api_url: str):
        self.api_url = api_url  # 例: https://xxxx.execute-api.ap-northeast-1.amazonaws.com/v1

    def upload_images(self, pil_images: list[Image.Image]) -> str | None:
        import io, json, requests

        photo_count = len(pil_images)
        payload = json.dumps({'photo_count': photo_count})
        headers = {'Content-Type': 'application/json'}

        # POST /upload
        response = requests.post(f"{self.api_url}/upload", data=payload, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        upload_urls = data['uploadUrls']
        viewer_url  = data['viewerUrl']

        # 署名付きURLへPUT
        for pil, url in zip(pil_images, upload_urls):
            buf = io.BytesIO()
            pil.save(buf, format='PNG')
            buf.seek(0)
            put_headers = {'Content-Type': 'image/png'}
            put_resp = requests.put(url, data=buf, headers=put_headers, timeout=30)
            put_resp.raise_for_status()

        return viewer_url


class CameraThread(threading.Thread):
    """カメラを裏で動かし続ける専門家"""
    def __init__(self, device_id: int = 0):
        super().__init__(daemon=True)
        self.device_id = device_id
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            print("!!! カメラエラー: カメラを開けませんでした。")
            return
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)
        if self.cap:
            self.cap.release()

    def get_latest_frame(self):
        """最新のカメラフレームを取得する（元仕様どおりのタイムアウト付き）"""
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            print("待機しましたが、カメラから新しいフレームを取得できませんでした。")
            return None

    def stop(self):
        self.stop_event.set()


# ==============================
# アプリ本体（customtkinter化）
# ==============================
class MainApplication(ctk.CTk):
    """アプリ全体の流れを管理する監督（UIをmodern化）"""

    def __init__(self):
        super().__init__()
        # ----- 外観テーマ -----
        ctk.set_appearance_mode("dark")        # "light" も可
        ctk.set_default_color_theme("blue")    # "green","dark-blue" 等

        self.title("プリクラアプリ ver1.2 (modern UI)")
        self.geometry("960x720")
        self.minsize(900, 680)

        # 変更禁止：API Gateway ベースURL
        API_GATEWAY_URL = "https://ezbu8f0gai.execute-api.ap-northeast-1.amazonaws.com"

        # 専門家の用意（責務は元の通り）
        self.camera = None  # カメラは撮影直前に起動
        self.asset_manager = AssetManager()
        self.editor = ImageEditor()
        self.compositor = ImageCompositor(self.editor)
        self.saver = ImageSaver()
        self.uploader = ImageUploader(API_GATEWAY_URL)
        self.captured_photos = []
        self.edited_photos = []

        # ルートレイアウト（カード式コンテナ）
        self.container = ctk.CTkFrame(self, corner_radius=16)
        self.container.pack(side="top", fill="both", expand=True, padx=16, pady=16)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # 画面を構成（Frameクラスはeditor_app側で定義）
        self.frames = {}
        frame_classes = (
            StartFrame, ShootingFrame, PhotoSelectionFrame, EditingFrame,
            PostEditChoiceFrame, FinalSelectionFrame, ThankYouFrame
        )
        for F in frame_classes:
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartFrame)

        # フッターバー（状態表示）
        self.status_var = ctk.StringVar(value="準備完了")
        self.status_bar = ctk.CTkLabel(self, textvariable=self.status_var, anchor="w")
        self.status_bar.pack(side="bottom", fill="x", padx=16, pady=(0, 12))

        # 終了処理
        self.protocol("WM_DELETE_WINDOW", self.shutdown)

    # -------- ナビゲーション --------
    def show_frame(self, frame_class):
        self.frames[frame_class].tkraise()

    # -------- 撮影開始 --------
    def start_camera(self):
        print("遷移: スタート -> 撮影画面")
        self.status_var.set("カメラ準備中…")

        # ステップ1：データ初期化（元仕様どおり）
        self.captured_photos.clear()
        self.edited_photos.clear()

        # ステップ2：カメラの専門家を新規起動
        self.camera = CameraCapture(CameraThread())
        self.camera.start()

        # UI遷移＆初期化
        self.after(500, self._open_shooting_frame)

    def _open_shooting_frame(self):
        sf = self.frames[ShootingFrame]
        sf.shoot_button.configure(state="disabled")
        sf.reset()
        self.show_frame(ShootingFrame)
        sf.update_camera_feed()
        self.status_var.set("撮影準備完了")

    def start_countdown(self, count: int = 3):
        shooting_frame = self.frames[ShootingFrame]
        shooting_frame.shoot_button.configure(state="disabled")
        if count > 0:
            shooting_frame.update_countdown(count)
            self.after(1000, lambda: self.start_countdown(count - 1))
        else:
            shooting_frame.update_countdown(0)
            self.capture_photo()
            self.after(500, lambda: shooting_frame.shoot_button.configure(state="normal"))

    def capture_photo(self):
        captured_frame = self.camera.capture()
        if captured_frame is not None:
            self.captured_photos.append(captured_frame)
            self.frames[ShootingFrame].update_photo_count(len(self.captured_photos))
            self.status_var.set(f"撮影済み: {len(self.captured_photos)} 枚")
        else:
            print("警告: フレームのキャプチャに失敗しました。")
            self.status_var.set("カメラからフレーム取得失敗")

    def go_to_photo_selection(self):
        all_photos = self.captured_photos + self.edited_photos
        if not all_photos:
            self.status_var.set("写真がありません")
            return
        self.frames[PhotoSelectionFrame].build_ui(all_photos)
        self.show_frame(PhotoSelectionFrame)

    def retake_photos(self):
        self.show_frame(ShootingFrame)

    def go_to_editing(self, selected_photo):
        # 元仕様：参照比較（NumPy配列）
        is_in_captured = any(np.array_equal(selected_photo, p) for p in self.captured_photos)
        is_in_edited = any(np.array_equal(selected_photo, p) for p in self.edited_photos)
        if is_in_captured:
            self.captured_photos = [p for p in self.captured_photos if not np.array_equal(p, selected_photo)]
        if is_in_edited:
            self.edited_photos = [p for p in self.edited_photos if not np.array_equal(p, selected_photo)]

        self.editor.set_photo(selected_photo)
        self.frames[EditingFrame].build_editor()
        self.show_frame(EditingFrame)

    def finish_editing(self):
        edited_image = self.compositor.get_final_image()
        if edited_image:
            cv_image = cv2.cvtColor(np.array(edited_image), cv2.COLOR_RGBA2BGRA)
            self.edited_photos.append(cv_image)
        self.show_frame(PostEditChoiceFrame)

    def go_to_final_selection(self):
        all_photos = self.captured_photos + self.edited_photos
        self.frames[FinalSelectionFrame].build_ui(all_photos)
        self.show_frame(FinalSelectionFrame)

    def upload_selected_photos(self):
        """選んだ写真をアップロード（AWS仕様不変）"""
        print("処理: 選択された写真をアップロード")
        photos_to_upload = self.frames[FinalSelectionFrame].get_selected_photos()
        if not photos_to_upload:
            print("アップロードする写真が選択されていません。")
            self.status_var.set("アップロード対象なし")
            return

        pil_images = []
        for cv2_image in photos_to_upload:
            image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            pil_images.append(pil_image)

        try:
            viewer_url = self.uploader.upload_images(pil_images)  # 既存メソッドをそのまま使用
        except Exception as e:
            print(f"エラー: アップロードに失敗しました: {e}")
            self.status_var.set("アップロード失敗")
            return

        if viewer_url:
            self.frames[ThankYouFrame].display_qr_code(viewer_url)
            self.show_frame(ThankYouFrame)
            self.status_var.set("アップロード完了")
        else:
            print("エラー: アップロードに失敗しました。")
            self.status_var.set("アップロード失敗")

    def go_to_start(self):
        print("遷移: サンキュー画面 -> スタート画面")
        self.show_frame(StartFrame)
        self.status_var.set("ホームに戻りました")

    def shutdown(self):
        """アプリ終了処理（元仕様を維持）"""
        print("アプリを終了します。")
        if self.camera:
            self.camera.stop()
        self.destroy()


if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
