"""影像增強模組：銳化、降噪、對比度調整。

採用保守參數設計，避免動漫線條過度處理而失去原本的乾淨感。
順序：bilateral filter 降噪 → unsharp mask 銳化 → CLAHE 對比增強。
"""
from __future__ import annotations
import cv2
import numpy as np

# === 增強參數（保守設定）===
# bilateral filter：保邊降噪
BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 30
BILATERAL_SIGMA_SPACE = 30

# unsharp mask：銳化強度
UNSHARP_SIGMA = 1.5
UNSHARP_AMOUNT = 0.3   # 銳化加成比例，越大線條越突出（保守設 0.3）

# CLAHE：局部對比度均衡化
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_SIZE = (8, 8)


def enhance_image(image: np.ndarray) -> np.ndarray:
    """對 RGB 圖片做銳化、降噪與對比度增強。

    Args:
        image: RGB 格式 numpy 陣列，形狀 (H, W, 3)。

    Returns:
        增強後的 RGB 圖片，形狀與輸入相同。
    """
    # OpenCV 操作以 BGR 為主，先轉色彩空間
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # === 1. 降噪：bilateral filter 保留邊緣 ===
    denoised = cv2.bilateralFilter(
        bgr,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE,
    )

    # === 2. 銳化：unsharp mask（原圖 - 模糊圖 → 加回原圖）===
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=UNSHARP_SIGMA)
    sharpened = cv2.addWeighted(
        denoised, 1.0 + UNSHARP_AMOUNT,
        blurred, -UNSHARP_AMOUNT,
        0,
    )

    # === 3. 對比度：在 LAB 色彩空間的 L 通道做 CLAHE，避免色偏 ===
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_SIZE,
    )
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 轉回 RGB 給下游處理
    return cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    import sys
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
    img = np.array(Image.open(test_path).convert("RGB"))
    enhanced = enhance_image(img)
    print(f"原圖：{img.shape}，dtype={img.dtype}")
    print(f"增強後：{enhanced.shape}，dtype={enhanced.dtype}")

    # 順便寫一張對照圖驗證效果（自動建立 outputs/）
    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    Image.fromarray(enhanced).save(out_dir / "_enhance_test.png")
    print(f"已輸出對照圖：{out_dir / '_enhance_test.png'}")
