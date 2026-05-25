"""sbhead-generater 主程式：Bang Dream 角色大頭貼自動生成系統。

執行方式：
    python main.py -i <圖片路徑>

流程：
    1. 驗證輸入檔案
    2. 讀圖（PIL → numpy RGB）
    3. 動漫臉部偵測（dghs-imgutils）
    4. 依臉部 bbox 裁切頭像
    5. 影像增強（銳化 + 降噪 + 對比）
    6. 判斷尺寸決定是否走 Real-ESRGAN 超解析度
    7. Resize 至 1024×1024 並輸出 PNG

輸出位置：main.py 同層的 outputs/ 資料夾，
檔名 output.png（衝突時自動編號 output(1).png、output(2).png ...）。
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image

from utils import (
    detect_largest_face,
    crop_by_bbox,
    enhance_image,
    upscale_image,
    save_avatar,
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


def main() -> None:
    args = parse_args()
    input_path: Path = args.input.resolve()

    print("================================================")
    print("  Bang Dream 角色大頭貼自動生成系統")
    print("================================================")
    print(f"輸入圖片：{input_path}")
    print()

    # === [1/6] 驗證輸入 ===
    print("[1/6] 驗證輸入圖片...")
    validate_input(input_path)

    # === [2/6] 讀圖（使用 PIL 避免中文路徑問題，轉成 numpy RGB）===
    print("[2/6] 讀取圖片...")
    image = np.array(Image.open(input_path).convert("RGB"))
    print(f"      原圖尺寸：{image.shape[1]}×{image.shape[0]}")

    # === [3/6] 偵測動漫臉部 ===
    print("[3/6] 偵測動漫角色臉部（dghs-imgutils）...")
    bbox = detect_largest_face(image)
    print(f"      最大臉部 bbox：{bbox}")

    # === [4/6] 依 bbox 裁切頭像（自動擴展含頭髮與肩膀）===
    print("[4/6] 裁切頭像區域...")
    cropped = crop_by_bbox(image, bbox)
    crop_size = cropped.shape[0]  # 已為正方形
    print(f"      裁切後尺寸：{crop_size}×{crop_size}")

    # === [5/6] 影像增強（銳化、降噪、對比度，永遠執行）===
    print("[5/6] 影像增強（銳化 + 降噪 + 對比）...")
    enhanced = enhance_image(cropped)

    # === [6/6] 超解析度或直接 resize ===
    if crop_size < OUTPUT_SIZE:
        print(
            f"[6/6] 裁切後尺寸 < {OUTPUT_SIZE}，"
            f"執行 Real-ESRGAN 超解析度放大（首次需載入模型，請稍候）..."
        )
        upscaled = upscale_image(enhanced)
    else:
        print(
            f"[6/6] 裁切後尺寸 ≥ {OUTPUT_SIZE}，"
            f"跳過 Real-ESRGAN，直接 resize..."
        )
        upscaled = enhanced

    # === 輸出 ===
    output_path = save_avatar(upscaled, OUTPUTS_DIR)

    print()
    print("================================================")
    print(f"完成！輸出至 {output_path}")
    print("================================================")


if __name__ == "__main__":
    main()
