"""sbhead-generater 主程式：Bang Dream 角色大頭貼自動生成系統。

執行方式：
    python main.py -i <圖片路徑>          # 單圖模式
    python main.py -i <資料夾路徑>        # 批次模式（遞迴掃描所有 JPG/JPEG/PNG）

流程：
    1. 驗證輸入（單檔或目錄）
    2. 讀圖（PIL → numpy RGB）
    3. 動漫臉部偵測（dghs-imgutils）：取得畫面上所有臉部 bbox
    4. 對每一張臉依序：裁切 → enhance → raw/sr 雙版本輸出
       - raw 版：直接 resize 4096
       - sr 版：< 4096 連續跑 Real-ESRGAN 直到 ≥ 4096；≥ 4096 直接 resize（與 raw 同）

輸出位置：main.py 同層的 outputs/raw/ 與 outputs/sr/ 兩個子資料夾。
檔名 output.png / output(1).png / output(2).png …，raw/ 與 sr/ 兩邊編號同步。
批次模式所有圖片的輸出全部打平到同一組 raw/ sr/ 資料夾，編號連續遞增。
單張臉處理失敗會印警告後跳過、繼續其他臉。
批次模式遇到無法讀取或無臉的圖片會印警告後跳過、繼續下一張圖。
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
OUTPUT_SIZE = 4096
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
        help="輸入圖片路徑（JPG / JPEG / PNG）或資料夾路徑（遞迴掃描）",
    )
    return parser.parse_args()


def validate_input(input_path: Path) -> None:
    """驗證輸入路徑存在；若為單檔則同時驗證格式。失敗時 sys.exit(1)。"""
    if not input_path.exists():
        print(f"錯誤：找不到路徑 {input_path}", file=sys.stderr)
        sys.exit(1)
    if input_path.is_file() and input_path.suffix.lower() not in SUPPORTED_EXTS:
        print(
            f"錯誤：不支援的格式 {input_path.suffix}，"
            f"只接受 JPG / JPEG / PNG。",
            file=sys.stderr,
        )
        sys.exit(1)


def collect_images(directory: Path) -> list[Path]:
    """遞迴掃描資料夾，回傳所有支援格式圖片路徑（字母排序）。"""
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def process_single_face(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """處理單一張臉，回傳 (raw_image, sr_image)。

    raw 版：裁切 → enhance（不跑 SR）
    sr  版：裁切 → enhance → 若 < 4096 連續跑 Real-ESRGAN 直到 ≥ 4096；≥ 4096 同 raw

    Args:
        image: 整張原圖（RGB）。
        bbox: 該張臉的 bbox (x0, y0, x1, y1)。

    Returns:
        (raw_image, sr_image)：兩張待寫入的 RGB 圖（尚未 resize 到 OUTPUT_SIZE）。
        後續由 save_paired_avatar 統一 resize 為 OUTPUT_SIZE × OUTPUT_SIZE 後輸出。
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


def run_single(input_path: Path) -> None:
    """單圖模式：處理一張圖片，無臉時 sys.exit(1)。"""
    print(f"輸入圖片：{input_path}")
    print()

    print("[1/3] 驗證輸入並讀取圖片...")
    image = np.array(Image.open(input_path).convert("RGB"))
    print(f"      原圖尺寸：{image.shape[1]}×{image.shape[0]}")

    print("[2/3] 偵測動漫角色臉部（dghs-imgutils）...")
    bboxes = detect_all_faces(image)
    if not bboxes:
        print("錯誤：圖片中未偵測到任何動漫角色臉部，無法生成大頭貼。", file=sys.stderr)
        sys.exit(1)
    n_faces = len(bboxes)
    print(f"      偵測到 {n_faces} 張臉")

    base_index = next_paired_index(OUTPUTS_DIR)
    print(f"[3/3] 依序處理 {n_faces} 張臉（起始編號 {base_index}）")
    print()

    success_paths: list[tuple[Path, Path]] = []
    failed_faces: list[int] = []

    for i, bbox in enumerate(bboxes):
        index = base_index + i
        print(f"  [{i + 1}/{n_faces}] 第 {i + 1} 張臉 bbox={bbox}")

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


def run_batch(image_paths: list[Path]) -> None:
    """批次模式：依序處理多張圖片，無臉或讀取失敗時警告+跳過。"""
    total_images = len(image_paths)
    print(f"批次模式：共找到 {total_images} 張圖片")
    print()

    next_index = next_paired_index(OUTPUTS_DIR)
    total_success = 0
    total_faces = 0
    skipped_images: list[str] = []

    for img_num, img_path in enumerate(image_paths, start=1):
        print(f"────────────────────────────────────────────────")
        print(f"[{img_num}/{total_images}] {img_path.name}")
        print(f"  路徑：{img_path}")

        # 讀圖
        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            print(f"  原圖尺寸：{image.shape[1]}×{image.shape[0]}")
        except Exception as exc:
            print(f"  警告：無法讀取圖片（{exc}），跳過。", file=sys.stderr)
            skipped_images.append(img_path.name)
            print()
            continue

        # 臉部偵測
        print(f"  偵測動漫角色臉部...")
        bboxes = detect_all_faces(image)
        if not bboxes:
            print(f"  警告：未偵測到任何臉部，跳過。", file=sys.stderr)
            skipped_images.append(img_path.name)
            print()
            continue

        n_faces = len(bboxes)
        total_faces += n_faces
        print(f"  偵測到 {n_faces} 張臉")

        # 處理每一張臉
        for i, bbox in enumerate(bboxes):
            print(f"  [{i + 1}/{n_faces}] bbox={bbox}")
            try:
                raw_image, sr_image = process_single_face(image, bbox)
                raw_path, sr_path = save_paired_avatar(
                    raw_image, sr_image, OUTPUTS_DIR, next_index,
                )
                print(f"        輸出 raw → {raw_path}")
                print(f"        輸出 sr  → {sr_path}")
                next_index += 1
                total_success += 1
            except Exception as exc:
                print(
                    f"        警告：第 {i + 1} 張臉處理失敗（{exc}），跳過。",
                    file=sys.stderr,
                )
        print()

    print("================================================")
    print(f"批次完成！")
    print(f"  處理圖片：{total_images} 張")
    print(f"  成功臉部：{total_success}/{total_faces} 張")
    if skipped_images:
        print(f"  跳過圖片（{len(skipped_images)} 張）：")
        for name in skipped_images:
            print(f"    {name}")
    print("================================================")


def main() -> None:
    args = parse_args()
    input_path: Path = args.input.resolve()

    print("================================================")
    print("  Bang Dream 角色大頭貼自動生成系統")
    print("================================================")

    validate_input(input_path)

    if input_path.is_dir():
        image_paths = collect_images(input_path)
        if not image_paths:
            print(
                f"錯誤：{input_path} 內找不到任何 JPG / JPEG / PNG 圖片。",
                file=sys.stderr,
            )
            sys.exit(1)
        run_batch(image_paths)
    else:
        run_single(input_path)


if __name__ == "__main__":
    main()
