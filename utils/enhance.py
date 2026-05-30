"""影像增強模組：降噪、銳化。

保守參數避免動漫線條過度處理。順序：bilateral 降噪 → unsharp 銳化。

備註：原本還有 CLAHE 局部對比度均衡，但動漫圖本身是高對比平塗，CLAHE 會在
臉頰等柔和漸層區把局部對比硬拉開，造成腮紅中心被壓暗、出現塊狀髒污（經 SR
放大後更明顯），收益最小、風險最大，故移除。對比度交由原圖既有調性即可。
"""
from __future__ import annotations
import cv2
import numpy as np

BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 30
BILATERAL_SIGMA_SPACE = 30

UNSHARP_SIGMA = 1.2
# 銳化力度。動漫線稿邊緣本身是抗鋸齒（柔和過渡像素）畫的，unsharp 會把邊緣
# 對比硬拉大、消掉抗鋸齒，露出像素階梯（鋸齒）。0.3 對乾淨線稿偏重，降到
# 0.15 在「提升清晰度」與「不破壞抗鋸齒」之間取平衡。
UNSHARP_AMOUNT = 0.15


def enhance_image(image: np.ndarray) -> np.ndarray:
    """對 RGB 圖片做降噪與銳化，回傳同形狀 RGB。"""
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

    return cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)


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
