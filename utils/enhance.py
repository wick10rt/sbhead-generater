from __future__ import annotations
import cv2
import numpy as np

BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 30
BILATERAL_SIGMA_SPACE = 30

UNSHARP_SIGMA = 1.2
# 0.15：銳化力度，再高會消掉動漫線稿抗鋸齒、露出鋸齒
UNSHARP_AMOUNT = 0.15


def enhance_image(image: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    denoised = cv2.bilateralFilter(
        bgr,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE,
    )

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
