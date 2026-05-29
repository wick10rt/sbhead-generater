"""大頭貼輸出模組（raw / sr 雙版本同步）。

每張臉輸出兩個 4096×4096 PNG：raw/（未做 SR）與 sr/（做完 SR），
兩邊用同一編號方便對比。命名 index=0 → output.png、index=N → output(N).png，
編號取兩邊現有最大值 + 1，不填補空缺。
"""
from __future__ import annotations
import re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

OUTPUT_SIZE = 4096
RAW_SUBDIR = "raw"
SR_SUBDIR = "sr"

_FILENAME_PATTERN = re.compile(r"^output(?:\((\d+)\))?\.png$")


def _index_to_filename(index: int) -> str:
    """index=0 → output.png；index=N → output(N).png。"""
    if index == 0:
        return "output.png"
    return f"output({index}).png"


def _max_index_in_dir(folder: Path) -> int:
    """回傳資料夾內現有最大編號（output.png 視為 0），無符合檔案回傳 -1。"""
    if not folder.exists():
        return -1

    max_num = -1
    for entry in folder.glob("output*.png"):
        match = _FILENAME_PATTERN.match(entry.name)
        if not match:
            continue
        num = int(match.group(1)) if match.group(1) else 0
        if num > max_num:
            max_num = num
    return max_num


def next_paired_index(outputs_dir: Path) -> int:
    """同時掃 raw/ 與 sr/，回傳兩邊最大編號 + 1；都空則回傳 0。"""
    raw_dir = outputs_dir / RAW_SUBDIR
    sr_dir = outputs_dir / SR_SUBDIR
    max_num = max(_max_index_in_dir(raw_dir), _max_index_in_dir(sr_dir))
    return max_num + 1


def _resize_to_output(image: np.ndarray) -> np.ndarray:
    """resize 到 OUTPUT_SIZE×OUTPUT_SIZE，縮小用 INTER_AREA、放大用 INTER_CUBIC。"""
    if image.shape[:2] == (OUTPUT_SIZE, OUTPUT_SIZE):
        return image
    h, w = image.shape[:2]
    interp = cv2.INTER_AREA if h * w > OUTPUT_SIZE * OUTPUT_SIZE else cv2.INTER_CUBIC
    return cv2.resize(image, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=interp)


def save_paired_avatar(
    raw_image: np.ndarray,
    sr_image: np.ndarray,
    outputs_dir: Path,
    index: int,
) -> tuple[Path, Path]:
    """將 raw 與 sr 兩張圖以同一編號輸出為 4096×4096 PNG，回傳兩個路徑。

    用 PIL 寫檔，避免 cv2.imwrite 在 Windows 中文路徑下失敗。
    """
    raw_dir = outputs_dir / RAW_SUBDIR
    sr_dir = outputs_dir / SR_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    sr_dir.mkdir(parents=True, exist_ok=True)

    filename = _index_to_filename(index)
    raw_path = raw_dir / filename
    sr_path = sr_dir / filename

    Image.fromarray(_resize_to_output(raw_image)).save(raw_path, format="PNG")
    Image.fromarray(_resize_to_output(sr_image)).save(sr_path, format="PNG")
    return raw_path, sr_path


if __name__ == "__main__":
    test_dir = Path(__file__).parent.parent / "outputs" / "_test_paired_naming"
    if test_dir.exists():
        for sub in (test_dir / RAW_SUBDIR, test_dir / SR_SUBDIR):
            if sub.exists():
                for f in sub.iterdir():
                    f.unlink()
                sub.rmdir()
        test_dir.rmdir()

    fake_raw = np.zeros((512, 512, 3), dtype=np.uint8)
    fake_sr = np.full((2048, 2048, 3), 200, dtype=np.uint8)

    results = []
    for _ in range(3):
        idx = next_paired_index(test_dir)
        raw_p, sr_p = save_paired_avatar(fake_raw, fake_sr, test_dir, idx)
        results.append((raw_p.name, sr_p.name))

    print("連續輸出三張臉的檔名測試：")
    for i, (r, s) in enumerate(results, start=1):
        print(f"第 {i} 張：raw={r} / sr={s}")

    expected = [
        ("output.png", "output.png"),
        ("output(1).png", "output(1).png"),
        ("output(2).png", "output(2).png"),
    ]
    print(f"預期：{expected}")
    print(f"實際：{results}")
    print(f"結果：{'通過' if results == expected else '失敗'}")
