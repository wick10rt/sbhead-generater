"""sbhead-generater 主程式：Bang Dream 角色大頭貼自動生成系統。

執行方式：
    python main.py -i <圖片路徑>    # 單圖
    python main.py -i <資料夾路徑>  # 批次（遞迴掃描 JPG/JPEG/PNG）
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
from utils.super_resolution import TARGET_SIZE as SR_TARGET_SIZE

# SR 觸發/目標尺寸（4096）。raw 與 sr 兩版最終都輸出 2048：raw 直接縮、
# sr 先超採樣到 ≥4096 再由 avatar_output 以 INTER_AREA 縮小（縮小即天然抗鋸齒）。
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}
PROJECT_ROOT = Path(__file__).parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    """解析命令列參數，只接受 -i / --input。"""
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
    """驗證輸入路徑存在；單檔則一併檢查格式，失敗時 sys.exit(1)。"""
    if not input_path.exists():
        print(f"錯誤：找不到路徑 {input_path}", file=sys.stderr)
        sys.exit(1)
    if input_path.is_file() and input_path.suffix.lower() not in SUPPORTED_EXTS:
        print(
            f"錯誤：不支援的格式 {input_path.suffix}，只接受 JPG / JPEG / PNG。",
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
    """處理單一張臉，回傳 (raw_image, sr_image)，皆尚未 resize 到輸出尺寸。

    raw 版只做裁切 + enhance；sr 版在裁切尺寸 < SR 目標（4096）時連續跑
    Real-ESRGAN 超採樣。最終輸出尺寸（raw/sr 皆 2048，sr 由超採樣 4096
    縮小）由 avatar_output.save_paired_avatar 統一處理。
    """
    cropped = crop_by_bbox(image, bbox)
    crop_size = cropped.shape[0]
    print(f"裁切完成，尺寸 {crop_size}×{crop_size}")

    enhanced = enhance_image(cropped)
    print("影像增強完成（銳化 + 降噪）")

    raw_image = enhanced

    # SR 觸發以「SR 目標尺寸」為準（非最終輸出尺寸）：裁切 < 4096 才需超採樣。
    if crop_size < SR_TARGET_SIZE:
        print(f"裁切尺寸 < {SR_TARGET_SIZE}，執行 Real-ESRGAN 超採樣")
        sr_image = upscale_image(enhanced)
    else:
        print(f"裁切尺寸 ≥ {SR_TARGET_SIZE}，跳過 SR（sr 版同 raw 版）")
        sr_image = enhanced

    return raw_image, sr_image


def run_single(input_path: Path) -> None:
    """單圖模式：無臉時 sys.exit(1)。"""
    print(f"輸入圖片：{input_path}")

    image = np.array(Image.open(input_path).convert("RGB"))
    print(f"原圖尺寸：{image.shape[1]}×{image.shape[0]}")

    print("偵測動漫角色臉部")
    bboxes = detect_all_faces(image)
    if not bboxes:
        print("錯誤：圖片中未偵測到任何動漫角色臉部，無法生成大頭貼。", file=sys.stderr)
        sys.exit(1)
    n_faces = len(bboxes)
    print(f"偵測到 {n_faces} 張臉")

    base_index = next_paired_index(OUTPUTS_DIR)
    print(f"依序處理 {n_faces} 張臉，起始編號 {base_index}")

    success_paths: list[tuple[Path, Path]] = []
    failed_faces: list[int] = []

    for i, bbox in enumerate(bboxes):
        index = base_index + i
        print(f"第 {i + 1}/{n_faces} 張臉 bbox={bbox}")
        try:
            raw_image, sr_image = process_single_face(image, bbox)
            raw_path, sr_path = save_paired_avatar(
                raw_image, sr_image, OUTPUTS_DIR, index,
            )
            print(f"輸出 raw → {raw_path}")
            print(f"輸出 sr → {sr_path}")
            success_paths.append((raw_path, sr_path))
        except Exception as exc:
            print(f"警告：第 {i + 1} 張臉處理失敗（{exc}），跳過。", file=sys.stderr)
            failed_faces.append(i + 1)

    print(f"完成，成功 {len(success_paths)}/{n_faces} 張")
    for raw_path, sr_path in success_paths:
        print(f"raw → {raw_path}")
        print(f"sr → {sr_path}")
    if failed_faces:
        print(f"失敗臉部編號：{failed_faces}")


def run_batch(image_paths: list[Path]) -> None:
    """批次模式：依序處理多張圖，無臉或讀取失敗時警告 + 跳過。"""
    total_images = len(image_paths)
    print(f"批次模式，共找到 {total_images} 張圖片")

    next_index = next_paired_index(OUTPUTS_DIR)
    total_success = 0
    total_faces = 0
    skipped_images: list[str] = []

    for img_num, img_path in enumerate(image_paths, start=1):
        print(f"第 {img_num}/{total_images} 張圖：{img_path}")

        try:
            image = np.array(Image.open(img_path).convert("RGB"))
            print(f"原圖尺寸：{image.shape[1]}×{image.shape[0]}")
        except Exception as exc:
            print(f"警告：無法讀取圖片（{exc}），跳過。", file=sys.stderr)
            skipped_images.append(img_path.name)
            continue

        print("偵測動漫角色臉部")
        bboxes = detect_all_faces(image)
        if not bboxes:
            print("警告：未偵測到任何臉部，跳過。", file=sys.stderr)
            skipped_images.append(img_path.name)
            continue

        n_faces = len(bboxes)
        total_faces += n_faces
        print(f"偵測到 {n_faces} 張臉")

        for i, bbox in enumerate(bboxes):
            print(f"第 {i + 1}/{n_faces} 張臉 bbox={bbox}")
            try:
                raw_image, sr_image = process_single_face(image, bbox)
                raw_path, sr_path = save_paired_avatar(
                    raw_image, sr_image, OUTPUTS_DIR, next_index,
                )
                print(f"輸出 raw → {raw_path}")
                print(f"輸出 sr → {sr_path}")
                next_index += 1
                total_success += 1
            except Exception as exc:
                print(f"警告：第 {i + 1} 張臉處理失敗（{exc}），跳過。", file=sys.stderr)

    print(f"批次完成，處理 {total_images} 張圖片，成功臉部 {total_success}/{total_faces} 張")
    if skipped_images:
        print(f"跳過 {len(skipped_images)} 張圖片：")
        for name in skipped_images:
            print(name)


def main() -> None:
    args = parse_args()
    input_path: Path = args.input.resolve()

    print("Bang Dream 角色大頭貼自動生成系統")

    validate_input(input_path)

    if input_path.is_dir():
        image_paths = collect_images(input_path)
        if not image_paths:
            print(f"錯誤：{input_path} 內找不到任何 JPG / JPEG / PNG 圖片。", file=sys.stderr)
            sys.exit(1)
        run_batch(image_paths)
    else:
        run_single(input_path)


if __name__ == "__main__":
    main()
