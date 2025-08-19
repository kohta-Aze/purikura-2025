# purikura-2025
This is a prototype of taking cute picture machine.I made this for the festival held by The University of reitaku in 2025.
# Purikura Booth (customtkinter版)

カメラで撮影 → お絵描き/スタンプ/フレーム編集 → 選択 → **AWS(API Gateway + S3 presigned URL)** にアップロード → **QRコード** を表示  
という一連の流れを、`customtkinter` ベースのモダンなUIで実現するデスクトップアプリです。

## 特長
- **機能**：撮影・編集・選択・AWSアップロード・QR表示
- **モダンUI**：`customtkinter` によるダーク/ライト対応、HiDPIで綺麗な表示（`CTkImage`採用）
- **責務分離**：`main_app.py`（オーケストレーション）と `editor_app.py`（UI/編集ロジック）に分割

---

## 動作環境
- **OS**: Windows 10/11（64bit）  
- **Python**: 3.10〜3.12（64bit 推奨）
- **カメラ**: 内蔵 or USB カメラ（OpenCV がアクセスできること）
- **ネットワーク**: アップロード時にインターネット接続が必要

---

## ディレクトリ構成（推奨）
project_root/
├─ main_app.py
├─ editor_app.py
├─ backgrounds/ # フレーム画像（PNG推奨, RGBA）
├─ stamps/ # スタンプ画像（PNG推奨, RGBA）
├─ build_windows.bat # EXEビルド用（任意）
├─ purikura.spec # PyInstaller用spec（任意）
└─ README.md
