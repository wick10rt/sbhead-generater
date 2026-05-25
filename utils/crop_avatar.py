"""依臉部 bbox 裁切大頭貼區域。

擴展策略：以臉部 bbox 的較長邊乘以 EXPAND_RATIO 作為正方形邊長，
中心向上偏移 EXTRA_TOP_RATIO 以包含更多頭髮。
若裁切框超出圖片邊界，會先嘗試平移保持尺寸，無法保持時再縮小。
"""
from __future__ import annotations
import numpy as np

# === 裁切擴展參數 ===
EXPAND_RATIO = 1.6       # 臉部 bbox 向外擴展倍數（含頭髮與肩膀）
EXTRA_TOP_RATIO = 0.15   # 上方額外擴展比例（將中心向上偏移以保留頭髮）


def crop_by_bbox(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """依臉部 bbox 擴展裁切出正方形大頭貼區域。

    Args:
        image: RGB 格式 numpy 陣列，形狀為 (H, W, 3)。
        bbox: 臉部 bbox，格式 (x0, y0, x1, y1)。

    Returns:
        裁切後的正方形 RGB 圖片。
    """
    h, w = image.shape[:2]
    x0, y0, x1, y1 = bbox

    # bbox 中心點與較長邊
    bbox_w = x1 - x0
    bbox_h = y1 - y0
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    # 正方形邊長 = 較長邊 × 擴展倍數
    side = max(bbox_w, bbox_h) * EXPAND_RATIO

    # 中心向上偏移，讓上方包含更多頭髮
    cy -= side * EXTRA_TOP_RATIO / 2.0

    # 計算正方形邊界（浮點數）
    sx0 = cx - side / 2.0
    sy0 = cy - side / 2.0
    sx1 = cx + side / 2.0
    sy1 = cy + side / 2.0

    # 若超出邊界，先嘗試平移保持正方形大小
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

    # 平移後仍可能超出（圖片本身比正方形小），最後 clip 並調整成正方形
    sx0 = max(0, int(round(sx0)))
    sy0 = max(0, int(round(sy0)))
    sx1 = min(w, int(round(sx1)))
    sy1 = min(h, int(round(sy1)))

    # 取較短邊作為最終正方形邊長
    final_side = min(sx1 - sx0, sy1 - sy0)
    sx1 = sx0 + final_side
    sy1 = sy0 + final_side

    return image[sy0:sy1, sx0:sx1].copy()


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from PIL import Image

    # 加入專案根目錄至 sys.path 以便 import 同層的 face_detect
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
    print(f"偵測到 {len(bboxes)} 張臉，依序裁切：")
    for i, bbox in enumerate(bboxes, start=1):
        cropped = crop_by_bbox(img, bbox)
        print(f"  第 {i} 張：bbox={bbox} → 裁切後 "
              f"{cropped.shape[1]}×{cropped.shape[0]}")
