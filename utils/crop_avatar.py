"""依臉部 bbox 裁切大頭貼區域。

以 bbox 較長邊 × EXPAND_RATIO 為正方形邊長，中心向上偏移 EXTRA_TOP_RATIO
以包含更多頭髮；超出邊界時先平移、無法保持尺寸再縮小。
"""
from __future__ import annotations
import numpy as np

EXPAND_RATIO = 1.6
EXTRA_TOP_RATIO = 0.15


def crop_by_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """依臉部 bbox 擴展裁切出正方形大頭貼區域。"""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox

    bbox_w = x1 - x0
    bbox_h = y1 - y0
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    side = max(bbox_w, bbox_h) * EXPAND_RATIO

    cy -= side * EXTRA_TOP_RATIO / 2.0

    sx0 = cx - side / 2.0
    sy0 = cy - side / 2.0
    sx1 = cx + side / 2.0
    sy1 = cy + side / 2.0

    if sx0 < 0:
        sx1 += -sx0
        sx0 = 0
    if sx1 > w:
        sx0 -= (sx1 - w)
        sx1 = w
    if sy0 < 0:
        sy1 += -sy0
        sy0 = 0
    if sy1 > h:
        sy0 -= (sy1 - h)
        sy1 = h

    sx0 = max(0, int(round(sx0)))
    sy0 = max(0, int(round(sy0)))
    sx1 = min(w, int(round(sx1)))
    sy1 = min(h, int(round(sy1)))

    final_side = min(sx1 - sx0, sy1 - sy0)
    sx1 = sx0 + final_side
    sy1 = sy0 + final_side

    return image[sy0:sy1, sx0:sx1].copy()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.face_detect import detect_all_faces

    sample_dir = Path(__file__).parent.parent / "sample_images"
    images = sorted(
        [p for p in sample_dir.iterdir()
         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not images:
        print(f"找不到測試圖，請放 JPG/PNG 到 {sample_dir}")
        sys.exit(1)

    test_path = images[0]
    img = np.array(Image.open(test_path).convert("RGB"))
    bboxes = detect_all_faces(img)
    print(f"原圖：{img.shape[1]}×{img.shape[0]}")
    print(f"偵測到 {len(bboxes)} 張臉，依序裁切")
    for i, bbox in enumerate(bboxes, start=1):
        cropped = crop_by_bbox(img, bbox)
        print(f"第 {i} 張：bbox={bbox} → 裁切後 {cropped.shape[1]}×{cropped.shape[0]}")
