"""
main_app.py
----------------------------------------
Purikuraアプリのエントリポイント。画面遷移（Start → Shooting → Selection → Editing → Final → QR）、
カメラ起動・撮影、編集結果の収集、AWS(S3)へのアップロードとQR表示までの
「アプリ全体の流れ（オーケストレーション）」を担当します。
UIは customtkinter ベースのモダンデザイン。

依存関係（機能分担）は editor_app.py 側の専門家クラス群に委譲します。
"""

import io
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests
from PIL import Image
import customtkinter as ctk

# editor_app 側の画面/専門家
from editor_app import (
    AssetManager, CameraCapture, ImageCompositor, ImageEditor, ImageSaver,
    EditingFrame, FinalSelectionFrame, PhotoSelectionFrame, PostEditChoiceFrame,
    ShootingFrame, StartFrame, ThankYouFrame
)
from ui.consent import request_share_consent
from ui.privacy_consent import PrivacyConsentManager


class ImageUploader:
    """
    AWSへ画像をアップロードする専門家（API Gateway + presigned URL）。

    概要
    ----
    1) API Gateway の `/upload` に POST して、アップロード用の presigned URL 群と viewerUrl を取得。
    2) 取得した各 URL に対して、画像を PUT でアップロードする。
    3) 成功したら viewerUrl を返す。

    Attributes
    ----------
    api_url : str
        API Gateway のベースURL（例: https://xxxx.execute-api.ap-northeast-1.amazonaws.com）

    Notes
    -----
    ・既存仕様を変更しないこと（エンドポイント・ペイロード・ヘッダ・PUTのContent-Type）。
    """

    def __init__(self, api_url: str):
        """
        コンストラクタ

        param api_url: API Gateway のベースURL
        """
        self.api_url = api_url

    def upload_images(self, pil_images: list[Image.Image]) -> str | None:
        """
        画像リストをS3へアップロードし、閲覧用URLを返す。

        関数の処理
        ----------
        - 画像枚数を含むJSONを `/upload` にPOSTして署名付きURL群を取得
        - 各URLへ画像(PNG)をPUT
        - viewerUrl を返却

        param pil_images: アップロードするPIL画像のリスト
        output: viewerUrl（文字列）。失敗時は None
        """
        photo_count = len(pil_images)
        payload = json.dumps({'photo_count': photo_count})
        headers = {'Content-Type': 'application/json'}

        # 1) 署名付きURL群とviewerUrlを取得
        response = requests.post(f"{self.api_url}/upload", data=payload, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        upload_urls = data['uploadUrls']
        viewer_url = data['viewerUrl']

        # 2) 各URLへPUTでアップロード
        for pil in pil_images:
            buf = io.BytesIO()
            pil.save(buf, format='PNG')
            buf.seek(0)
            put_headers = {'Content-Type': 'image/png'}
            resp = requests.put(upload_urls.pop(0), data=buf, headers=put_headers, timeout=30)
            resp.raise_for_status()

        # 3) 閲覧URLを返す
        return viewer_url


class CameraThread(threading.Thread):
    """
    カメラをバックグラウンドで回し続けるスレッド。

    概要
    ----
    OpenCV で定期的にフレームを取得し、最新1枚だけをキューに保持。
    取り出し側は「最新を1秒待ちで取得」できる。

    Attributes
    ----------
    device_id : int
        OpenCV のカメラデバイスID
    frame_queue : queue.Queue
        最新フレームを格納するキュー（常に最大1件）
    stop_event : threading.Event
        スレッド停止の合図
    cap : cv2.VideoCapture | None
        実カメラハンドル
    """

    def __init__(self, device_id: int = 0):
        """
        コンストラクタ

        param device_id: OpenCV の VideoCapture デバイス番号（既定0）
        """
        super().__init__(daemon=True)
        self.device_id = device_id
        self.frame_queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.cap = None

    def run(self):
        """
        スレッド本体
        キューに最新フレームを詰め続ける。取り出しは get_latest_frame() で行う。
        """
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
        """
        最新のカメラフレームを1秒待ちで取得する。

        関数の処理
        ----------
        - キューから最新フレームを取り出す（最大1秒待ち）
        - 取れなければ None を返す

        output: OpenCV画像（ndarray）または None
        """
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            print("待機しましたが、カメラから新しいフレームを取得できませんでした。")
            return None

    def stop(self):
        """
        スレッド停止の合図を送る。

        関数の処理
        ----------
        - stop_event を set して run() のループ終了を促す
        """
        self.stop_event.set()


class MainApplication(ctk.CTk):
    """
    アプリ全体の司令塔（UIルートウィンドウ）。

    役割
    ----
    ・画面（Frame）を生成して切替管理
    ・カメラ開始/撮影/編集/最終選択/アップロード/QR表示の一連の流れを制御

    Attributes
    ----------
    camera : CameraCapture | None
        撮影中に使うカメラ管理クラス。開始時に初期化される
    asset_manager : AssetManager
        フレーム/スタンプ素材の読み込みを担当
    editor : ImageEditor
        編集用のレイヤ状態を保持
    compositor : ImageCompositor
        レイヤ合成の担当
    saver : ImageSaver
        画像保存の担当（本アプリでは任意）
    uploader : ImageUploader
        AWSアップロードの担当
    captured_photos : list[np.ndarray]
        撮影直後の写真（OpenCV配列）
    edited_photos : list[np.ndarray]
        編集後に確定した写真（OpenCV配列）
    frames : dict[type, ctk.CTkFrame]
        画面インスタンスの辞書
    status_var : ctk.StringVar
        ステータスバーの文言
    """

    def __init__(self):
        """
        ウィンドウ初期化と各コンポーネントの準備。

        関数の処理
        ----------
        - テーマ設定、画面生成、ステータスバー作成
        - editor_app側の画面クラスをインスタンス化してカードスタック化
        - 終了ハンドラ登録
        """
        super().__init__()
        # 外観テーマ
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("プリクラアプリ ver1.2 (modern UI)")
        self.geometry("960x720")
        self.minsize(900, 680)

        # API Gateway ベースURL（仕様固定）
        API_GATEWAY_URL = "https://ezbu8f0gai.execute-api.ap-northeast-1.amazonaws.com"

        self.samples_dir = Path('samples')
        self.samples_dir.mkdir(exist_ok=True)
        self.sample_index_path = self.samples_dir / "index.json"
        self._ensure_sample_index()
        self.ar_settings: dict[str, object] = {
            'enable_crown': True,
            'enable_aura': True,
            'ema_alpha': 0.2,
            'aura_color': (255, 120, 220),
            'aura_intensity': 0.6,
            'aura_frequency': 1.5,
            'aura_spread': 2.0,
            'aura_mix_mode': 'rgb_add',
        }

        # 専門家インスタンス
        self.camera = None
        self.asset_manager = AssetManager()
        self.editor = ImageEditor()
        self.compositor = ImageCompositor(self.editor)
        self.saver = ImageSaver()
        self.uploader = ImageUploader(API_GATEWAY_URL)
        self.captured_photos: list[np.ndarray] = []
        self.edited_photos: list[np.ndarray] = []

        # ルートコンテナ
        self.container = ctk.CTkFrame(self, corner_radius=16)
        self.container.pack(side="top", fill="both", expand=True, padx=16, pady=16)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        # 画面の登録
        self.frames: dict[type, ctk.CTkFrame] = {}
        frame_classes = (
            StartFrame, ShootingFrame, PhotoSelectionFrame, EditingFrame,
            PostEditChoiceFrame, FinalSelectionFrame, ThankYouFrame
        )
        for F in frame_classes:
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.consent_manager = PrivacyConsentManager(self)

        self.show_frame(StartFrame)

        # ステータスバー
        self.status_var = ctk.StringVar(value="準備完了")
        self.status_bar = ctk.CTkLabel(self, textvariable=self.status_var, anchor="w")
        self.status_bar.pack(side="bottom", fill="x", padx=16, pady=(0, 12))

        self.after(100, self.consent_manager.ensure_consent)

        # 終了処理
        self.protocol("WM_DELETE_WINDOW", self.shutdown)

    # ---------- 画面制御 ----------
    def show_frame(self, frame_class):
        """
        指定の画面を最前面に表示する。

        param frame_class: 表示したいFrameクラス（StartFrame等）
        """
        frame = self.frames[frame_class]
        frame.tkraise()
        if hasattr(frame, 'on_show'):
            frame.on_show()

    def update_ar_settings(self, settings: dict[str, object]):
        """AR設定を更新し、カメラへ反映する。"""

        self.ar_settings.update(settings)
        if self.camera:
            self.camera.configure_ar(self.ar_settings)

    def _ensure_sample_index(self):
        """samples/index.json が存在しなければ初期化する。"""

        if not self.sample_index_path.exists():
            with self.sample_index_path.open('w', encoding='utf-8') as fp:
                json.dump([], fp, ensure_ascii=False, indent=2)

    def _sanitize_metadata(self, metadata: list[dict[str, object]] | None) -> list[dict[str, object]]:
        """JSONシリアライズ可能な形に変換する。"""

        sanitized: list[dict[str, object]] = []
        if not metadata:
            return sanitized
        for action in metadata:
            safe_action: dict[str, object] = {}
            for key, value in action.items():
                if isinstance(value, tuple):
                    safe_action[key] = list(value)
                else:
                    safe_action[key] = value
            sanitized.append(safe_action)
        return sanitized

    def register_sample(self, image: Image.Image, metadata: list[dict[str, object]] | None):
        """共有許諾済み画像を samples/ に保存しインデックスを更新する。"""

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filename = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image_copy = image.copy()
        image_copy.thumbnail((640, 480))
        image_path = self.samples_dir / filename
        image_copy.save(image_path, "PNG")

        with self.sample_index_path.open('r', encoding='utf-8') as fp:
            try:
                index_data = json.load(fp)
            except json.JSONDecodeError:
                index_data = []
        entry = {
            'image': filename,
            'meta': self._sanitize_metadata(metadata),
            'savedAt': timestamp,
        }
        index_data.insert(0, entry)
        with self.sample_index_path.open('w', encoding='utf-8') as fp:
            json.dump(index_data, fp, ensure_ascii=False, indent=2)
        start_frame = self.frames[StartFrame]
        start_frame.add_sample_entry(entry)

    def _prompt_share_consent(self, image: Image.Image, metadata: list[dict[str, object]] | None):
        """共有許諾モーダルを開き、結果に応じて保存する。"""

        def _callback(allow: bool):
            if allow:
                self.register_sample(image, metadata)

        request_share_consent(self, _callback, preview_image=image)

    # ---------- 撮影 ----------
    def start_camera(self):
        """
        撮影を開始する（スタート画面のボタンから呼ばれる）。

        利用同意が得られていなければダイアログを表示して処理を保留し、
        同意後に撮影準備へ進む。
        """
        manager = getattr(self, "consent_manager", None)
        if manager:
            if manager.is_declined:
                return
            if not manager.ensure_consent(self._start_camera_after_consent):
                return

        self._start_camera_after_consent()

    def _start_camera_after_consent(self):
        """同意取得後に実行する撮影準備処理。"""
        print("遷移: スタート -> 撮影画面")
        self.status_var.set("カメラ準備中…")

        # 1) 既存データのクリア
        self.captured_photos.clear()
        self.edited_photos.clear()

        # 2) カメラ起動
        self.camera = CameraCapture(CameraThread())
        self.camera.configure_ar(self.ar_settings)
        self.camera.start()

        # 3) 画面遷移（少し待ってから）
        self.after(500, self._open_shooting_frame)

    def _open_shooting_frame(self):
        """
        撮影画面へ切り替え、UIを初期化してプレビュー更新を開始する。
        """
        sf = self.frames[ShootingFrame]
        sf.shoot_button.configure(state="disabled")
        sf.reset()
        self.show_frame(ShootingFrame)
        sf.update_camera_feed()
        self.status_var.set("撮影準備完了")

    def start_countdown(self, count: int = 3):
        """
        撮影のカウントダウンを開始する。

        param count: カウント秒（既定3）
        """
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
        """
        現在のフレームを1枚キャプチャして撮影済みリストに加える。
        """
        captured_frame = self.camera.capture()
        if captured_frame is not None:
            self.captured_photos.append(captured_frame)
            self.frames[ShootingFrame].update_photo_count(len(self.captured_photos))
            self.status_var.set(f"撮影済み: {len(self.captured_photos)} 枚")
        else:
            print("警告: フレームのキャプチャに失敗しました。")
            self.status_var.set("カメラからフレーム取得失敗")

    # ---------- 編集/選択 ----------
    def go_to_photo_selection(self):
        """
        撮影済み＋編集済みの写真一覧画面へ遷移する。
        """
        all_photos = self.captured_photos + self.edited_photos
        if not all_photos:
            self.status_var.set("写真がありません")
            return
        self.frames[PhotoSelectionFrame].build_ui(all_photos)
        self.show_frame(PhotoSelectionFrame)

    def retake_photos(self):
        """
        撮影画面へ戻る（再撮影）。
        """
        self.show_frame(ShootingFrame)

    def go_to_editing(self, selected_photo):
        """
        選択された写真を編集画面へ渡す。

        param selected_photo: OpenCV配列の写真（一覧からの選択）
        """
        # 参照比較で、一覧から重複を除いた上で編集対象に設定
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
        """
        編集結果を確定し、編集済みリストに追加して次画面へ。
        """
        edited_image = self.compositor.get_final_image()
        if edited_image:
            metadata = self.editor.get_edit_history()
            self._prompt_share_consent(edited_image.copy(), metadata)
            cv_image = cv2.cvtColor(np.array(edited_image), cv2.COLOR_RGBA2BGRA)
            self.edited_photos.append(cv_image)
        self.show_frame(PostEditChoiceFrame)

    def go_to_final_selection(self):
        """
        最終選択画面へ遷移し、アップロード対象をチェック選択させる。
        """
        all_photos = self.captured_photos + self.edited_photos
        self.frames[FinalSelectionFrame].build_ui(all_photos)
        self.show_frame(FinalSelectionFrame)

    # ---------- アップロード/終了 ----------
    def upload_selected_photos(self):
        """
        選択された写真をAWSにアップロードして、QR画面へ遷移する。

        関数の処理
        ----------
        - FinalSelectionFrame から選択写真を取得
        - OpenCV → PIL に変換
        - ImageUploader.upload_images() でS3にPUT
        - 成功時は ThankYouFrame に viewerUrl のQRを表示
        """
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
            viewer_url = self.uploader.upload_images(pil_images)
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
        """
        スタート画面へ戻る。
        """
        print("遷移: サンキュー画面 -> スタート画面")
        self.show_frame(StartFrame)
        self.status_var.set("ホームに戻りました")

    def shutdown(self):
        """
        アプリ終了処理。

        関数の処理
        ----------
        - カメラ稼働中なら停止
        - ウィンドウ破棄
        """
        print("アプリを終了します。")
        if self.camera:
            self.camera.stop()
        self.destroy()


if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
