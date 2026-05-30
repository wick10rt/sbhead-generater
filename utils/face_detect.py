from __future__ import annotations
import numpy as np
from PIL import Image
from imgutils.detect import detect_faces


def detect_all_faces(
    image: np.ndarray,
) -> list[tuple[int, int, int, int]]:
    pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image

    detections = detect_faces(pil_image)

    return [
        (int(x0), int(y0), int(x1), int(y1))
        for (x0, y0, x1, y1), _label, _score in detections
    ]


if __name__ == "__main__":
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

    bboxes = detect_all_faces(img)
    print(f"偵測到 {len(bboxes)} 張臉")
    for i, (x0, y0, x1, y1) in enumerate(bboxes, start=1):
        print(f"第 {i} 張：bbox=({x0}, {y0}, {x1}, {y1})，尺寸={x1 - x0}×{y1 - y0}")
