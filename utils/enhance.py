"""影像增強模組：降噪、銳化、對比度。

保守參數避免動漫線條過度處理。順序：bilateral 降噪 → unsharp 銳化 → CLAHE 對比。
"""
from __future__ import annotations
import cv2
import numpy as np

BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 30
BILATERAL_SIGMA_SPACE = 30

UNSHARP_SIGMA = 1.5
UNSHARP_AMOUNT = 0.3

CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_SIZE = (8, 8)


def enhance_image(image: np.ndarray) -> np.ndarray:
    """對 RGB 圖片做降噪、銳化與對比度增強，回傳同形狀 RGB。"""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    denoised = cv2.bilateralFilter(
        bgr,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE,
    )

    # unsharp mask：原圖加權減去模糊圖
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=UNSHARP_SIGMA)
    sharpened = cv2.addWeighted(
        denoised, 1.0 + UNSHARP_AMOUNT,
        blurred, -UNSHARP_AMOUNT,
        0,
    )

    # 只在 LAB 的 L 通道做 CLAHE，避免色偏
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_SIZE,
    )
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge([l_channel, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

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

    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    Image.fromarray(enhanced).save(out_dir / "_enhance_test.png")
    print(f"已輸出對照圖：{out_dir / '_enhance_test.png'}")
