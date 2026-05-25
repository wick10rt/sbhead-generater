"""動漫角色臉部偵測模組。

使用 dghs-imgutils 偵測圖片中所有動漫角色臉部，回傳面積最大者的 bbox。
若沒有偵測到任何臉部，直接 sys.exit(1) 終止程式。
"""
from __future__ import annotations
import sys
import numpy as np
from PIL import Image
from imgutils.detect import detect_faces


def detect_largest_face(image: np.ndarray) -> tuple[int, int, int, int]:
    """偵測 RGB 圖片中所有動漫臉部，回傳最大臉部的 bbox。

    Args:
        image: RGB 格式 numpy 陣列，形狀為 (H, W, 3)。

    Returns:
        最大臉部 bbox，格式 (x0, y0, x1, y1)，皆為 int。

    Notes:
        若未偵測到任何臉部，會印出錯誤訊息並 sys.exit(1)，不會回傳。
    """
    # dghs-imgutils 0.19+ 的 detect_faces 不接受 ndarray，
    # 只認 PIL.Image / 檔案路徑 / BinaryIO，所以先轉 PIL.Image。
    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image

    # detect_faces 回傳 list，元素為 ((x0, y0, x1, y1), label, score)
    detections = detect_faces(pil_image)

    if not detections:
        print(
            "錯誤：圖片中未偵測到任何動漫角色臉部，無法生成大頭貼。",
            file=sys.stderr,
        )
        sys.exit(1)

    # 以 bbox 面積排序，取最大者
    def _area(detection) -> int:
        x0, y0, x1, y1 = detection[0]
        return (x1 - x0) * (y1 - y0)

    largest_bbox = max(detections, key=_area)[0]
    x0, y0, x1, y1 = largest_bbox
    return (int(x0), int(y0), int(x1), int(y1))


if __name__ == "__main__":
    # 單獨測試：讀 sample_images/ 內第一張圖
    from pathlib import Path
    from PIL import Image

    sample_dir = Path(__file__).parent.parent / "sample_images"
    images = sorted(
        [p for p in sample_dir.iterdir()
         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not images:
        print(f"找不到測試圖，請放 JPG/PNG 到 {sample_dir}")
        sys.exit(1)

    test_path = images[0]
    print(f"測試圖：{test_path}")

    img = np.array(Image.open(test_path).convert("RGB"))
    print(f"圖片尺寸：{img.shape[1]}×{img.shape[0]}")

    bbox = detect_largest_face(img)
    print(f"最大臉部 bbox：{bbox}")
    print(f"臉部尺寸：{bbox[2] - bbox[0]}×{bbox[3] - bbox[1]}")
