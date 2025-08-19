"""
editor_app.py
----------------------------------------
UIの各画面（Frame）と、素材管理・編集レイヤ・合成・保存・カメラ表示など
「専門家」クラス群を定義するモジュール。

customtkinter を用いたモダンUI実装。
HiDPI対応のため、画像表示には CTkImage を使用します。
"""

import os
from datetime import datetime
import cv2
from PIL import Image, ImageDraw
import qrcode
import customtkinter as ctk


# =========================================================
# コア機能（カメラ/アセット/編集/合成/保存）
# =========================================================
class CameraCapture:
    """
    カメラからの映像取得を高レベルに管理するラッパー。

    役割
    ----
    - バックグラウンドの CameraThread と対話し、最新フレームの取得や開始/停止を提供
    - UIで使える縮小済みの CTkImage を生成

    Attributes
    ----------
    thread : CameraThread
        実際にフレームを供給するスレッド
    """

    def __init__(self, camera_thread):
        """
        コンストラクタ

        param camera_thread: 起動済み/未起動の CameraThread インスタンス
        """
        self.thread = camera_thread

    def start(self):
        """
        スレッドを開始する。

        関数の処理
        ----------
        - CameraThread.start() を呼ぶ
        """
        self.thread.start()

    def stop(self):
        """
        スレッドを停止する。

        関数の処理
        ----------
        - CameraThread.stop() を呼ぶ
        """
        self.thread.stop()

    def capture(self) -> cv2.typing.MatLike | None:
        """
        最新フレーム（OpenCV配列）を1枚取得する。

        output: 画像配列 or None（取得できない場合）
        """
        return self.thread.get_latest_frame()

    def get_display_frame(self) -> ctk.CTkImage | None:
        """
        ライブビュー表示用の縮小 CTkImage を返す（HiDPI対応）。

        関数の処理
        ----------
        - 最新フレームをRGBに変換 → PIL化 → サムネイル化
        - CTkImage(light/dark同一) を生成して返す

        output: ctk.CTkImage or None
        """
        frame = self.thread.get_latest_frame()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img.thumbnail((960, 540))
            return ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        return None


class AssetManager:
    """
    画像素材（背景フレーム・スタンプ）を読み込んで保持する専門家。

    Attributes
    ----------
    backgrounds : list[dict]
        {'name': ファイル名, 'image': PIL.Image} の配列
    stamps : list[dict]
        同上（スタンプ用）
    """

    def __init__(self, bg_folder: str = 'backgrounds', stamp_folder: str = 'stamps'):
        """
        コンストラクタ

        param bg_folder: フレーム画像のディレクトリ
        param stamp_folder: スタンプ画像のディレクトリ
        """
        print("AssetManager: 背景フレームを読み込んでいます...")
        self.backgrounds = self._load_images(bg_folder)
        print("AssetManager: スタンプを読み込んでいます...")
        self.stamps = self._load_images(stamp_folder)

    def _load_images(self, folder: str) -> list[dict]:
        """
        指定フォルダから画像を読み込み、RGBAにして返す。

        param folder: 読み込み対象ディレクトリ
        output: [{'name': str, 'image': PIL.Image}, ...]
        """
        if not os.path.exists(folder):
            print(f"警告: '{folder}' フォルダが見つかりません。")
            return []
        images = []
        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image_path = os.path.join(folder, filename)
                    image = Image.open(image_path).convert("RGBA")
                    images.append({'name': filename, 'image': image})
                except Exception as e:
                    print(f"画像読み込みエラー: {filename} - {e}")
        return images


class ImageEditor:
    """
    編集対象画像の「レイヤ状態（photo/frame/drawing）」を保持する専門家。

    Attributes
    ----------
    photo_layer : PIL.Image | None
        元写真（RGBA）
    frame_layer : PIL.Image | None
        適用中のフレーム（写真サイズにリサイズ済み）
    drawing_layer : PIL.Image | None
        ユーザーの落書き・スタンプを乗せる透明レイヤ
    """

    def __init__(self):
        """コンストラクタ：空のレイヤを初期化"""
        self.photo_layer: Image.Image | None = None
        self.frame_layer: Image.Image | None = None
        self.drawing_layer: Image.Image | None = None

    def set_photo(self, cv2_image: cv2.typing.MatLike):
        """
        編集対象の写真を設定する。

        関数の処理
        ----------
        - BGR(OpenCV) → RGB → PIL(RGBA) 化
        - drawing_layer を同サイズの透明キャンバスで初期化

        param cv2_image: OpenCV画像
        """
        original_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        self.photo_layer = Image.fromarray(original_image).convert("RGBA")
        self.drawing_layer = Image.new("RGBA", self.photo_layer.size, (0, 0, 0, 0))

    def set_frame(self, frame_image: Image.Image):
        """
        フレーム画像を現在の写真サイズに合わせて設定する。

        param frame_image: フレーム（RGBA推奨）
        """
        if self.photo_layer:
            self.frame_layer = frame_image.resize(self.photo_layer.size)

    def get_drawing_canvas(self) -> Image.Image | None:
        """
        現在の描画レイヤ（PIL.Image）を返す。

        output: PIL.Image or None
        """
        return self.drawing_layer


class ImageCompositor:
    """
    レイヤ（写真・フレーム・描画）を1枚に合成する専門家。
    """

    def __init__(self, editor: ImageEditor):
        """
        コンストラクタ

        param editor: ImageEditor インスタンス（レイヤ状態の保持元）
        """
        self.editor = editor

    def create_final_image(self) -> Image.Image | None:
        """
        レイヤを合成して最終画像を生成する。

        関数の処理
        ----------
        - photo_layer をベースに frame_layer, drawing_layer を順に alpha 合成

        output: 合成済みPIL.Image（RGBA） or None
        """
        if not self.editor.photo_layer:
            return None
        composite = self.editor.photo_layer.copy()
        if self.editor.frame_layer:
            composite = Image.alpha_composite(composite, self.editor.frame_layer)
        if self.editor.drawing_layer:
            composite = Image.alpha_composite(composite, self.editor.drawing_layer)
        return composite

    def get_final_image(self) -> Image.Image | None:
        """
        create_final_image() のエイリアス（互換API維持）。
        """
        return self.create_final_image()


class ImageSaver:
    """
    画像をファイルに保存する専門家。
    """

    def save(self, image: Image.Image, path: str = 'output', filename: str = None) -> str:
        """
        PNGで保存し、保存パスを返す。

        関数の処理
        ----------
        - 保存先ディレクトリを作成
        - ファイル名未指定時はタイムスタンプで自動命名
        - PNGで保存

        param image: 保存したいPIL.Image
        param path:  出力ディレクトリ
        param filename: 出力ファイル名（省略時は自動）
        output: 保存先のフルパス（str）
        """
        if not os.path.exists(path):
            os.makedirs(path)
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"purikura_{timestamp}.png"
        save_path = os.path.join(path, filename)
        image.save(save_path, "PNG")
        print(f"画像を {save_path} に保存しました。")
        return save_path


# =========================================================
# UIの部品（セレクタ/画面）
# =========================================================
class _ScrollableThumbs(ctk.CTkScrollableFrame):
    """
    サムネイルをグリッド配置で並べる共通部品。

    Attributes
    ----------
    columns : int
        1行あたりの列数
    _row, _col : int
        現在のグリッド位置（内部管理）
    """

    def __init__(self, master, columns=4, **kwargs):
        """
        param master: 親ウィジェット
        param columns: グリッド列数
        """
        super().__init__(master, **kwargs)
        self.columns = columns
        self._row = 0
        self._col = 0

    def add_thumb(self, image_pil: Image.Image, command):
        """
        サムネイルボタンを1つ追加する。

        関数の処理
        ----------
        - 画像を縮小し CTkImage 化
        - 画像付きCTkButtonを配置し、押下時に command を実行
        - グリッド座標を自動で進める

        param image_pil: PIL.Image（原寸）
        param command: クリック時に呼ぶコールバック
        """
        img = image_pil.copy()
        img.thumbnail((120, 120))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        btn = ctk.CTkButton(
            self, image=ctk_img, text="", width=130, height=130,
            fg_color="transparent", hover_color=("gray85", "gray25"),
            command=command
        )
        btn.image = ctk_img  # 参照保持
        btn.grid(row=self._row, column=self._col, padx=8, pady=8, sticky="nsew")

        self._col += 1
        if self._col >= self.columns:
            self._col = 0
            self._row += 1


class BackgroundSelector(ctk.CTkToplevel):
    """
    フレーム（背景）画像を選択するポップアップ。
    """

    def __init__(self, parent, assets: list[dict], callback):
        """
        param parent: 親ウィンドウ
        param assets: [{'name': str, 'image': PIL.Image}, ...]
        param callback: 選択完了時に呼ぶ関数（引数 image: PIL.Image）
        """
        super().__init__(parent)
        self.title("フレーム選択")
        self.geometry("680x520")
        self.callback = callback

        ctk.CTkLabel(self, text="お好きなフレームを選択してください").pack(pady=(12, 6))
        scroll = _ScrollableThumbs(self, columns=5)
        scroll.pack(fill="both", expand=True, padx=12, pady=12)
        for asset in assets:
            scroll.add_thumb(asset['image'], command=lambda a=asset: self._select(a))

    def _select(self, asset: dict):
        """
        選ばれた画像を親へ返してクローズ。

        param asset: {'name': str, 'image': PIL.Image}
        """
        self.callback(asset['image'])
        self.destroy()


class StampSelector(ctk.CTkToplevel):
    """
    スタンプ画像を選択するポップアップ。
    """

    def __init__(self, parent, assets: list[dict], callback):
        """
        param parent: 親ウィンドウ
        param assets: [{'name': str, 'image': PIL.Image}, ...]
        param callback: 選択完了時に呼ぶ関数（引数 image: PIL.Image）
        """
        super().__init__(parent)
        self.title("スタンプ選択")
        self.geometry("680x520")
        self.callback = callback

        ctk.CTkLabel(self, text="配置したいスタンプを選択してください").pack(pady=(12, 6))
        scroll = _ScrollableThumbs(self, columns=6)
        scroll.pack(fill="both", expand=True, padx=12, pady=12)
        for asset in assets:
            scroll.add_thumb(asset['image'], command=lambda a=asset: self._select(a))

    def _select(self, asset: dict):
        """
        選ばれたスタンプを親へ返してクローズ。

        param asset: {'name': str, 'image': PIL.Image}
        """
        self.callback(asset['image'])
        self.destroy()


# ---------------- 画面（Frame群） ----------------
class StartFrame(ctk.CTkFrame):
    """
    撮影スタート画面。
    役割：アプリのウェルカム表示と「撮影開始」ボタンのみ。
    """

    def __init__(self, parent, controller):
        """
        param parent: 親コンテナ
        param controller: MainApplication（画面遷移に利用）
        """
        super().__init__(parent)
        self.controller = controller

        header = ctk.CTkLabel(self, text="Purikura Booth", font=ctk.CTkFont(size=36, weight="bold"))
        header.pack(pady=(24, 6))

        sub = ctk.CTkLabel(self, text="おしゃれプリクラ体験へようこそ", font=ctk.CTkFont(size=16))
        sub.pack(pady=(0, 18))

        big_start = ctk.CTkButton(
            self, text="撮影をはじめる", height=56, width=280,
            font=ctk.CTkFont(size=20, weight="bold"),
            command=self.controller.start_camera
        )
        big_start.pack(pady=8)

        hint = ctk.CTkLabel(self, text="※カメラと素材の読み込みには数秒かかることがあります")
        hint.pack(pady=(6, 0))


class ShootingFrame(ctk.CTkFrame):
    """
    撮影画面。
    役割：ライブビュー表示、カウントダウン撮影、撮影枚数表示、選択画面への遷移。
    """

    def __init__(self, parent, controller):
        """
        param parent: 親コンテナ
        param controller: MainApplication
        """
        super().__init__(parent)
        self.controller = controller

        # ライブビュー
        self.preview_label = ctk.CTkLabel(self, text="カメラ接続中…", width=800, height=450, corner_radius=12)
        self.preview_label.pack(pady=(12, 6))

        # カウントダウン表示
        self.count_var = ctk.StringVar(value="")
        ctk.CTkLabel(self, textvariable=self.count_var, font=ctk.CTkFont(size=48, weight="bold")).pack()

        # 操作ボタン
        btn_row = ctk.CTkFrame(self)
        btn_row.pack(pady=10)
        self.shoot_button = ctk.CTkButton(btn_row, text="3秒カウントで撮影", width=200,
                                          command=lambda: controller.start_countdown(3))
        self.shoot_button.grid(row=0, column=0, padx=6)
        ctk.CTkButton(btn_row, text="選択へ", width=140,
                      command=controller.go_to_photo_selection).grid(row=0, column=1, padx=6)
        ctk.CTkButton(btn_row, text="やりなおす", width=140,
                      command=controller.retake_photos).grid(row=0, column=2, padx=6)

        # 撮影枚数
        self.count_taken_var = ctk.StringVar(value="0 枚")
        ctk.CTkLabel(self, text="撮影済み：", font=ctk.CTkFont(size=14)).pack()
        ctk.CTkLabel(self, textvariable=self.count_taken_var).pack()

        self._after_id = None

    def reset(self):
        """
        画面状態を初期化（カウント表示/タイマ停止/枚数リセット）。
        """
        self.count_var.set("")
        self.count_taken_var.set("0 枚")
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None

    def update_camera_feed(self):
        """
        ライブビュー画像を定期更新する。

        関数の処理
        ----------
        - CameraCapture から CTkImage を取得してプレビューに反映
        - 初回フレーム取得時に撮影ボタンを有効化
        - 約33msごとに再スケジューリング（~30fps）
        """
        frame_img = self.controller.camera.get_display_frame() if self.controller.camera else None
        if frame_img:
            self.preview_label.configure(image=frame_img, text="")
            self.preview_label.image = frame_img  # 参照保持
            if str(self.shoot_button.cget("state")) == "disabled":
                self.shoot_button.configure(state="normal")
        self._after_id = self.after(33, self.update_camera_feed)

    def update_countdown(self, n: int):
        """
        カウントダウン表示を更新。

        param n: 残り秒数（0 以下で「ぱしゃ！」表示）
        """
        self.count_var.set(f"{n}" if n > 0 else "ぱしゃ！")

    def update_photo_count(self, n: int):
        """
        撮影済み枚数表示を更新。

        param n: 撮影済み枚数
        """
        self.count_taken_var.set(f"{n} 枚")


class PhotoSelectionFrame(ctk.CTkFrame):
    """
    撮影済み/編集済みの写真から、編集へ進める1枚を選ぶ画面。
    """

    def __init__(self, parent, controller):
        """
        param parent: 親コンテナ
        param controller: MainApplication
        """
        super().__init__(parent)
        self.controller = controller
        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text="写真を選んで編集へ", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(12, 6))
        self.scroll = _ScrollableThumbs(self, columns=5)
        self.scroll.pack(fill="both", expand=True, padx=16, pady=12)

        nav = ctk.CTkFrame(self)
        nav.pack(pady=(0, 12))
        ctk.CTkButton(nav, text="撮影に戻る", command=controller.retake_photos, width=140).grid(row=0, column=0, padx=6)
        ctk.CTkButton(nav, text="最終選択へ", command=controller.go_to_final_selection, width=140).grid(row=0, column=1, padx=6)

    def build_ui(self, photos_cv: list[cv2.typing.MatLike]):
        """
        写真の一覧UIを構築する。

        関数の処理
        ----------
        - 既存のサムネイル行をクリア
        - 各OpenCV画像をPIL化 → サムネイル生成 → クリックで編集へ

        param photos_cv: OpenCV画像のリスト
        """
        for w in list(self.scroll.children.values()):
            w.destroy()
        self.scroll._row = self.scroll._col = 0

        for cvimg in photos_cv:
            rgb = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            self.scroll.add_thumb(pil, command=lambda p=cvimg: self.controller.go_to_editing(p))


class EditingFrame(ctk.CTkFrame):
    """
    編集画面。描画・フレーム選択・スタンプ配置・確定を行う。
    """

    def __init__(self, parent, controller):
        """
        param parent: 親コンテナ
        param controller: MainApplication
        """
        super().__init__(parent)
        self.controller = controller

        # 左：プレビュー
        self.preview_label = ctk.CTkLabel(self, text="画像を読み込み中…", width=760, height=520, corner_radius=12)
        self.preview_label.grid(row=0, column=0, rowspan=3, padx=(16, 8), pady=16, sticky="nsew")

        # 右：ツール群
        tool_panel = ctk.CTkFrame(self, corner_radius=12)
        tool_panel.grid(row=0, column=1, padx=(8, 16), pady=(16, 8), sticky="nsew")

        ctk.CTkLabel(tool_panel, text="ツール", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(12, 6))

        # ペン選択
        pen_row = ctk.CTkFrame(tool_panel)
        pen_row.pack(pady=6, fill="x")
        self.pen_color = "#FF69B4"  # 現在のペン色
        self.pen_width = 10         # 現在のペン太さ

        def pen_btn(text, color):
            return ctk.CTkButton(pen_row, text=text, width=120, command=lambda c=color: self.change_pen_color(c))
        pen_btn("キラキラピンク", "#FF69B4").grid(row=0, column=0, padx=4, pady=4)
        pen_btn("キラキラブルー", "#00BFFF").grid(row=0, column=1, padx=4, pady=4)
        pen_btn("くっきりブラック", "black").grid(row=1, column=0, padx=4, pady=4)
        pen_btn("けしゴム", "eraser").grid(row=1, column=1, padx=4, pady=4)

        ctk.CTkButton(tool_panel, text="ぜんぶ消す", command=self.clear_all_drawings).pack(pady=(6, 2))

        # アセット
        asset_row = ctk.CTkFrame(tool_panel)
        asset_row.pack(pady=(12, 8))
        ctk.CTkButton(asset_row, text="フレーム選択", width=120, command=self.open_frame_selector).grid(row=0, column=0, padx=4, pady=4)
        ctk.CTkButton(asset_row, text="スタンプ", width=120, command=self.open_stamp_selector).grid(row=0, column=1, padx=4, pady=4)

        # 決定
        ctk.CTkButton(tool_panel, text="これで決定！", height=44, command=self.controller.finish_editing).pack(pady=(10, 12), fill="x", padx=8)

        # 操作ヒント
        ctk.CTkLabel(tool_panel, text="プレビュー上でドラッグしてお絵描き。\nスタンプモード時はクリックで配置。").pack(pady=(0, 12))

        # レイアウト比率
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 編集状態
        self.is_drawing = False
        self.last_x, self.last_y = None, None
        self.display_image: ctk.CTkImage | None = None
        self.display_size: tuple[int, int] | None = None  # CTkImage表示サイズを保持

        # イベント
        self.preview_label.bind("<B1-Motion>", self.draw_on_canvas)
        self.preview_label.bind("<ButtonRelease-1>", self.stop_drawing)
        self.preview_label.bind("<Button-1>", self.place_stamp_at_click)

    # --- 既存ロジック互換（プロパティ） ---
    @property
    def editor(self): return self.controller.editor
    @property
    def compositor(self): return self.controller.compositor
    @property
    def asset_manager(self): return self.controller.asset_manager

    def build_editor(self):
        """
        編集開始時にプレビューを初期表示する。
        """
        self.update_display_image()

    def change_pen_color(self, new_color: str):
        """
        ペン色（もしくは消しゴム/スタンプモード）を切り替える。

        param new_color: "#RRGGBB" / "black" / "eraser" / "stamp_mode"
        """
        self.pen_color = new_color
        self.pen_width = 50 if new_color == "eraser" else 10

    def clear_all_drawings(self):
        """
        描画レイヤを完全にクリアする。
        """
        if self.editor.photo_layer:
            self.editor.drawing_layer = Image.new("RGBA", self.editor.photo_layer.size, (0, 0, 0, 0))
            self.update_display_image()

    def set_frame(self, frame_image: Image.Image):
        """
        フレームを適用してプレビュー更新。

        param frame_image: フレームPIL画像
        """
        self.editor.set_frame(frame_image)
        self.update_display_image()

    def draw_on_canvas(self, event):
        """
        プレビュー上でドラッグした軌跡を描画レイヤへ反映する。

        関数の処理
        ----------
        - 表示画像サイズから元画像サイズへのスケール変換
        - ペン/消しゴム/キラキラ（擬似ブラシ）を描画

        param event: Tkのマウスイベント（event.x, event.y を使用）
        """
        if not self.is_drawing:
            self.is_drawing = True
            self.last_x, self.last_y = event.x, event.y
            return
        if not self.editor.drawing_layer or not self.editor.photo_layer:
            return

        draw = ImageDraw.Draw(self.editor.drawing_layer)
        w, h = self.editor.photo_layer.size
        # CTkImageにはwidth()/height()が無いので、保存したサイズを利用
        disp_w, disp_h = (self.display_size if self.display_size else self.editor.photo_layer.size)

        def scale(x, y): return (x * w / disp_w, y * h / disp_h)

        scaled_last = scale(self.last_x, self.last_y)
        scaled_current = scale(event.x, event.y)

        if self.pen_color == "eraser":
            draw.line([scaled_last, scaled_current], fill=(0, 0, 0, 0), width=self.pen_width, joint="round")
        elif self.pen_color == "black":
            draw.line([scaled_last, scaled_current], fill=self.pen_color, width=self.pen_width, joint="round")
        else:
            self.draw_sparkly_line(draw, scaled_last, scaled_current)

        self.last_x, self.last_y = event.x, event.y
        self.update_display_image()

    def draw_sparkly_line(self, draw: ImageDraw.ImageDraw, start: tuple, end: tuple):
        """
        キラキラ系の擬似ブラシ。メインラインの周囲にランダムなスパーク点を打つ。

        param draw: ImageDraw.Draw（描画先）
        param start: 始点 (x, y) in 元画像座標
        param end:   終点 (x, y) in 元画像座標
        """
        import random as _r
        draw.line([start, end], fill=self.pen_color, width=self.pen_width, joint="round")
        num_sparks = int(((start[0] - end[0])**2 + (start[1] - end[1])**2)**0.5 / 2)
        for _ in range(num_sparks):
            r = _r.random()
            x = start[0] * (1 - r) + end[0] * r
            y = start[1] * (1 - r) + end[1] * r
            dx = _r.uniform(-self.pen_width * 2, self.pen_width * 2)
            dy = _r.uniform(-self.pen_width * 2, self.pen_width * 2)
            draw.point((x + dx, y + dy), fill=_r.choice([self.pen_color, "white"]))

    def stop_drawing(self, event):
        """
        マウスボタンを離したら描画状態を終了する。
        """
        self.is_drawing = False
        self.last_x, self.last_y = None, None

    def place_stamp_at_click(self, event):
        """
        スタンプモード時、クリック位置にスタンプを貼る。

        関数の処理
        ----------
        - 表示 → 元画像座標へスケール変換
        - スタンプ中心がクリック位置に来るように貼付

        param event: Tkのマウスイベント
        """
        if self.pen_color != "stamp_mode":
            return
        if hasattr(self, 'current_stamp') and self.editor.photo_layer and self.editor.drawing_layer:
            w, h = self.editor.photo_layer.size
            disp_w, disp_h = (self.display_size if self.display_size else self.editor.photo_layer.size)
            stamp_w, stamp_h = self.current_stamp.size
            top_left_x = int(event.x * w / disp_w) - stamp_w // 2
            top_left_y = int(event.y * h / disp_h) - stamp_h // 2
            self.editor.drawing_layer.paste(self.current_stamp, (top_left_x, top_left_y), mask=self.current_stamp)
            self.update_display_image()

    def open_stamp_selector(self):
        """
        スタンプ選択ポップアップを開く。
        """
        if not self.asset_manager.stamps:
            print("スタンプ用画像がありません。")
            return
        StampSelector(self, self.asset_manager.stamps, self.activate_stamp_mode)

    def activate_stamp_mode(self, stamp_image: Image.Image):
        """
        選んだスタンプを現在のスタンプとして登録し、モードを有効化。

        param stamp_image: スタンプ画像（PIL）
        """
        self.pen_color = "stamp_mode"
        self.current_stamp = stamp_image.copy()
        self.current_stamp.thumbnail((150, 150))

    def update_display_image(self):
        """
        合成結果をリサイズしてプレビューへ反映（CTkImage化）。
        """
        if not self.editor.photo_layer:
            return
        preview_image = self.compositor.create_final_image()
        if preview_image:
            resized = preview_image.copy()
            resized.thumbnail((960, 540))
            self.display_image = ctk.CTkImage(light_image=resized, dark_image=resized, size=resized.size)
            self.display_size = resized.size  # (w, h)
            self.preview_label.configure(image=self.display_image, text="")
            self.preview_label.image = self.display_image  # 参照保持

    def open_frame_selector(self):
        """
        フレーム選択ポップアップを開く。
        """
        if not self.asset_manager.backgrounds:
            print("フレーム用画像がありません。")
            return
        BackgroundSelector(self, self.asset_manager.backgrounds, self.set_frame)


class PostEditChoiceFrame(ctk.CTkFrame):
    """
    編集直後の分岐画面。
    役割：編集を続ける/撮影へ戻る/最終選択へ から次を選択させる。
    """

    def __init__(self, parent, controller):
        """
        param parent: 親コンテナ
        param controller: MainApplication
        """
        super().__init__(parent)
        self.controller = controller

        ctk.CTkLabel(self, text="次の操作を選んでください", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=(16, 8))

        row = ctk.CTkFrame(self)
        row.pack(pady=10)
        ctk.CTkButton(row, text="さらに編集する", width=180, command=lambda: controller.show_frame(EditingFrame)).grid(row=0, column=0, padx=6)
        ctk.CTkButton(row, text="撮影に戻る", width=160, command=controller.retake_photos).grid(row=0, column=1, padx=6)
        ctk.CTkButton(row, text="最終選択へ", width=160, command=controller.go_to_final_selection).grid(row=0, column=2, padx=6)


class FinalSelectionFrame(ctk.CTkFrame):
    """
    アップロード対象の写真をチェックボックスで選ぶ画面。
    """

    def __init__(self, parent, controller):
        """
        param parent: 親コンテナ
        param controller: MainApplication
        """
        super().__init__(parent)
        self.controller = controller
        self.selected: list[cv2.typing.MatLike] = []

        ctk.CTkLabel(self, text="アップロードする写真を選んでください", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=(16, 8))
        self.scroll = ctk.CTkScrollableFrame(self, height=420, corner_radius=12)
        self.scroll.pack(fill="both", expand=True, padx=16, pady=12)

        nav = ctk.CTkFrame(self)
        nav.pack(pady=(0, 12))
        ctk.CTkButton(nav, text="撮影に戻る", command=controller.retake_photos, width=140).grid(row=0, column=0, padx=6)
        ctk.CTkButton(nav, text="アップロード", command=controller.upload_selected_photos, width=160).grid(row=0, column=1, padx=6)

    def build_ui(self, photos_cv: list[cv2.typing.MatLike]):
        """
        一覧UIを構築し、各アイテムにチェックボックスを付ける。

        param photos_cv: OpenCV画像のリスト
        """
        for w in list(self.scroll.children.values()):
            w.destroy()
        self.selected = []

        def add_item(cvimg):
            rgb = cv2.cvtColor(cvimg, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            pil_thumb = pil.copy()
            pil_thumb.thumbnail((220, 220))
            ctk_img = ctk.CTkImage(light_image=pil_thumb, dark_image=pil_thumb, size=pil_thumb.size)

            row = ctk.CTkFrame(self.scroll, corner_radius=8)
            row.pack(fill="x", padx=8, pady=6)

            img_label = ctk.CTkLabel(row, image=ctk_img, text="")
            img_label.image = ctk_img  # 参照保持
            img_label.pack(side="left", padx=8, pady=8)

            var = ctk.BooleanVar(value=False)

            def on_toggle():
                if var.get():
                    self.selected.append(cvimg)
                else:
                    try:
                        self.selected.remove(cvimg)
                    except ValueError:
                        pass

            ctk.CTkCheckBox(row, text="この写真を選ぶ", variable=var, command=on_toggle).pack(side="left", padx=12)

        for cvimg in photos_cv:
            add_item(cvimg)

    def get_selected_photos(self) -> list[cv2.typing.MatLike]:
        """
        チェックされた写真のOpenCV配列リストを返す。

        output: list[np.ndarray]
        """
        return list(self.selected)


class ThankYouFrame(ctk.CTkFrame):
    """
    アップロード完了とQRコード表示画面。
    """

    def __init__(self, parent, controller):
        """
        param parent: 親コンテナ
        param controller: MainApplication
        """
        super().__init__(parent)
        self.controller = controller

        ctk.CTkLabel(self, text="アップロード完了！QRを読み取ってください", font=ctk.CTkFont(size=22, weight="bold")).pack(pady=(16, 8))
        self.qr_canvas = ctk.CTkLabel(self, text="")
        self.qr_canvas.pack(pady=12)

        nav = ctk.CTkFrame(self)
        nav.pack(pady=10)
        ctk.CTkButton(nav, text="ホームへ戻る", width=160, command=controller.go_to_start).grid(row=0, column=0, padx=6)

    def display_qr_code(self, url: str):
        """
        渡されたURLをQRコード化して画面に表示する。

        関数の処理
        ----------
        - qrcode でQR生成 → PIL化 → 320pxにリサイズ
        - CTkImage化してラベルに貼り付け

        param url: 閲覧ページのURL
        """
        qr = qrcode.QRCode(box_size=8, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        img = img.resize((320, 320))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.qr_canvas.configure(image=ctk_img, text="")
        self.qr_canvas.image = ctk_img  # 参照保持
