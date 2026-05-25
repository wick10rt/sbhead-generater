"""大頭貼輸出模組。

固定輸出 1024×1024 PNG 至指定資料夾，自動建立資料夾與自動編號避免覆寫。
命名規則：output.png → output(1).png → output(2).png → ...
規則為「永遠用現存最大編號 + 1」（不填補編號空缺）。
"""
from __future__ import annotations
import re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

OUTPUT_SIZE = 1024

# 比對 output.png 或 output(N).png
_FILENAME_PATTERN = re.compile(r"^output(?:\((\d+)\))?\.png$")


def _next_output_path(outputs_dir: Path) -> Path:
    """掃描資料夾，回傳下一個可用檔名（最大編號 + 1）。

    output.png 視為編號 0；output(N).png 視為編號 N。
    若資料夾內沒有任何 output*.png，回傳 output.png。

    Args:
        outputs_dir: 輸出資料夾路徑（不存在會自動建立）。

    Returns:
        下一個可用的完整檔案路徑。
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)

    max_num = -1  # -1 代表還沒有任何 output.png
    for entry in outputs_dir.glob("output*.png"):
        match = _FILENAME_PATTERN.match(entry.name)
        if not match:
            continue
        num = int(match.group(1)) if match.group(1) else 0
        if num > max_num:
            max_num = num

    if max_num == -1:
        return outputs_dir / "output.png"
    return outputs_dir / f"output({max_num + 1}).png"


def save_avatar(image: np.ndarray, outputs_dir: Path) -> Path:
    """將 RGB 圖片 resize 到 1024×1024 並輸出 PNG。

    使用 PIL 寫檔，避免 cv2.imwrite 在 Windows 中文路徑下失敗。

    Args:
        image: RGB 格式 numpy 陣列。
        outputs_dir: 輸出資料夾路徑（會自動建立）。

    Returns:
        實際輸出的檔案路徑。
    """
    # 確保最終尺寸為 1024×1024
    if image.shape[:2] != (OUTPUT_SIZE, OUTPUT_SIZE):
        # 縮小用 INTER_AREA、放大用 INTER_CUBIC
        h, w = image.shape[:2]
        if h * w > OUTPUT_SIZE * OUTPUT_SIZE:
            interp = cv2.INTER_AREA
        else:
            interp = cv2.INTER_CUBIC
        image = cv2.resize(
            image, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=interp,
        )

    output_path = _next_output_path(outputs_dir)
    Image.fromarray(image).save(output_path, format="PNG")
    return output_path


if __name__ == "__main__":
    # 測試自動編號邏輯
    test_dir = Path(__file__).parent.parent / "outputs" / "_test_naming"
    # 清空測試資料夾
    if test_dir.exists():
        for f in test_dir.iterdir():
            f.unlink()
        test_dir.rmdir()

    fake = np.zeros((1024, 1024, 3), dtype=np.uint8)
    p1 = save_avatar(fake, test_dir)
    p2 = save_avatar(fake, test_dir)
    p3 = save_avatar(fake, test_dir)
    print(f"連續輸出檔名測試：")
    print(f"  第 1 張：{p1.name}")
    print(f"  第 2 張：{p2.name}")
    print(f"  第 3 張：{p3.name}")
    expected = ["output.png", "output(1).png", "output(2).png"]
    actual = [p1.name, p2.name, p3.name]
    print(f"\n預期：{expected}")
    print(f"實際：{actual}")
    print(f"結果：{'通過' if actual == expected else '失敗'}")
