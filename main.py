"""sbhead-generater 主程式：Bang Dream 角色大頭貼自動生成系統。

執行方式：
    python main.py -i <圖片路徑>

流程：
    1. 驗證輸入檔案
    2. 讀圖（PIL → numpy RGB）
    3. 動漫臉部偵測（dghs-imgutils）：取得畫面上所有臉部 bbox
    4. 對每一張臉依序：裁切 → enhance → raw/sr 雙版本輸出
       - raw 版：直接 resize 1024
       - sr 版：< 1024 跑 Real-ESRGAN；≥ 1024 直接 resize（與 raw 同）

輸出位置：main.py 同層的 outputs/raw/ 與 outputs/sr/ 兩個子資料夾。
檔名 output.png / output(1).png / output(2).png …，
raw/ 與 sr/ 兩邊編號同步；單張臉處理失敗會印警告後跳過、繼續其他臉。
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image

from utils import (
    detect_all_faces,
    crop_by_bbox,
    enhance_image,
    upscale_image,
    next_paired_index,
    save_paired_avatar,
)

# === 全域設定 ===
OUTPUT_SIZE = 1024
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path(__file__).parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    """解析命令列參數。只接受 -i / --input 一個必填參數。"""
    parser = argparse.ArgumentParser(
        prog="sbhead-generater",
        description="Bang Dream 角色大頭貼自動生成系統",
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        type=Path,
        help="輸入圖片路徑（支援 JPG / JPEG / PNG）",
    )
    return parser.parse_args()


def validate_input(input_path: Path) -> None:
    """驗證輸入檔案存在且格式支援；失敗時 sys.exit(1)。"""
    if not input_path.exists():
        print(f"錯誤：找不到輸入檔案 {input_path}", file=sys.stderr)
        sys.exit(1)
    if not input_path.is_file():
        print(f"錯誤：{input_path} 不是檔案", file=sys.stderr)
        sys.exit(1)
    if input_path.suffix.lower() not in SUPPORTED_EXTS:
        print(
            f"錯誤：不支援的格式 {input_path.suffix}，"
            f"只接受 JPG / JPEG / PNG。",
            file=sys.stderr,
        )
        sys.exit(1)


def process_single_face(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """處理單一張臉，回傳 (raw_image, sr_image)。

    raw 版：裁切 → enhance（不跑 SR）
    sr  版：裁切 → enhance → 若 < 1024 跑 Real-ESRGAN；≥ 1024 同 raw

    Args:
        image: 整張原圖（RGB）。
        bbox: 該張臉的 bbox (x0, y0, x1, y1)。

    Returns:
        (raw_image, sr_image)：兩張待寫入的 RGB 圖（尚未 resize 到 1024）。
        後續由 save_paired_avatar 統一 resize 為 1024×1024 後輸出。
    """
    # [1/4] 裁切
    cropped = crop_by_bbox(image, bbox)
    crop_size = cropped.shape[0]  # 已為正方形
    print(f"        [1/4] 裁切完成，尺寸 {crop_size}×{crop_size}")

    # [2/4] enhance（raw 與 sr 共用同一份增強結果）
    enhanced = enhance_image(cropped)
    print(f"        [2/4] 影像增強完成（銳化 + 降噪 + 對比）")

    # [3/4] raw 版：不跑 SR，直接拿增強後的結果
    raw_image = enhanced
    print(f"        [3/4] raw 版備妥（未做 SR）")

    # [4/4] sr 版：依裁切尺寸決定是否跑 Real-ESRGAN
    if crop_size < OUTPUT_SIZE:
        print(f"        [4/4] 裁切尺寸 < {OUTPUT_SIZE}，執行 Real-ESRGAN...")
        sr_image = upscale_image(enhanced)
    else:
        print(f"        [4/4] 裁切尺寸 ≥ {OUTPUT_SIZE}，跳過 SR（sr 版同 raw 版）")
        sr_image = enhanced

    return raw_image, sr_image


def main() -> None:
    args = parse_args()
    input_path: Path = args.input.resolve()

    print("================================================")
    print("  Bang Dream 角色大頭貼自動生成系統")
    print("================================================")
    print(f"輸入圖片：{input_path}")
    print()

    # === [1/3] 驗證輸入 + 讀圖 ===
    print("[1/3] 驗證輸入並讀取圖片...")
    validate_input(input_path)
    image = np.array(Image.open(input_path).convert("RGB"))
    print(f"      原圖尺寸：{image.shape[1]}×{image.shape[0]}")

    # === [2/3] 偵測所有臉部 ===
    print("[2/3] 偵測動漫角色臉部（dghs-imgutils）...")
    bboxes = detect_all_faces(image)
    n_faces = len(bboxes)
    print(f"      偵測到 {n_faces} 張臉")

    # === [3/3] 對每一張臉做 raw / sr 雙版本輸出 ===
    base_index = next_paired_index(OUTPUTS_DIR)
    print(f"[3/3] 依序處理 {n_faces} 張臉（起始編號 {base_index}）")
    print()

    success_paths: list[tuple[Path, Path]] = []
    failed_faces: list[int] = []

    for i, bbox in enumerate(bboxes):
        index = base_index + i
        face_label = f"[{i + 1}/{n_faces}] 第 {i + 1} 張臉"
        print(f"  {face_label} bbox={bbox}")

        try:
            raw_image, sr_image = process_single_face(image, bbox)
            raw_path, sr_path = save_paired_avatar(
                raw_image, sr_image, OUTPUTS_DIR, index,
            )
            print(f"        輸出 raw → {raw_path}")
            print(f"        輸出 sr  → {sr_path}")
            success_paths.append((raw_path, sr_path))
        except Exception as exc:
            print(
                f"        警告：第 {i + 1} 張臉處理失敗（{exc}），跳過。",
                file=sys.stderr,
            )
            failed_faces.append(i + 1)
        print()

    # === 收尾總結 ===
    print("================================================")
    print(f"完成！成功 {len(success_paths)}/{n_faces} 張")
    if success_paths:
        print("輸出檔案：")
        for raw_path, sr_path in success_paths:
            print(f"  raw → {raw_path}")
            print(f"  sr  → {sr_path}")
    if failed_faces:
        print(f"失敗臉部編號：{failed_faces}")
    print("================================================")


if __name__ == "__main__":
    main()
