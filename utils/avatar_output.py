from __future__ import annotations
import re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

OUTPUT_SIZE = 2048
RAW_SUBDIR = "raw"
SR_SUBDIR = "sr"

_FILENAME_PATTERN = re.compile(r"^output(?:\((\d+)\))?\.png$")


def _index_to_filename(index: int) -> str:
    if index == 0:
        return "output.png"
    return f"output({index}).png"


def _max_index_in_dir(folder: Path) -> int:
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
    max_num = max(
        _max_index_in_dir(outputs_dir / RAW_SUBDIR),
        _max_index_in_dir(outputs_dir / SR_SUBDIR),
    )
    return max_num + 1


def _resize_to(image: np.ndarray, size: int) -> np.ndarray:
    if image.shape[:2] == (size, size):
        return image
    h, w = image.shape[:2]
    interp = cv2.INTER_AREA if h * w > size * size else cv2.INTER_CUBIC
    return cv2.resize(image, (size, size), interpolation=interp)


def save_paired_avatar(
    raw_image: np.ndarray,
    sr_image: np.ndarray,
    outputs_dir: Path,
    index: int,
) -> tuple[Path, Path]:
    raw_dir = outputs_dir / RAW_SUBDIR
    sr_dir = outputs_dir / SR_SUBDIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    sr_dir.mkdir(parents=True, exist_ok=True)

    filename = _index_to_filename(index)
    raw_path = raw_dir / filename
    sr_path = sr_dir / filename

    Image.fromarray(_resize_to(raw_image, OUTPUT_SIZE)).save(raw_path, format="PNG")
    Image.fromarray(_resize_to(sr_image, OUTPUT_SIZE)).save(sr_path, format="PNG")
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
    fake_sr = np.full((4096, 4096, 3), 200, dtype=np.uint8)

    results = []
    sizes = []
    for _ in range(3):
        idx = next_paired_index(test_dir)
        raw_p, sr_p = save_paired_avatar(fake_raw, fake_sr, test_dir, idx)
        results.append((raw_p.name, sr_p.name))
        if not sizes:
            sizes = [Image.open(raw_p).size, Image.open(sr_p).size]

    print("連續輸出三張臉的檔名測試：")
    for i, (r, s) in enumerate(results, start=1):
        print(f"第 {i} 張：raw={r} / sr={s}")

    expected = [
        ("output.png", "output.png"),
        ("output(1).png", "output(1).png"),
        ("output(2).png", "output(2).png"),
    ]
    print(f"預期檔名：{expected}")
    print(f"實際檔名：{results}")
    print(f"尺寸（raw / sr）：{sizes}，預期 [(2048,2048),(2048,2048)]")
    name_ok = results == expected
    size_ok = sizes == [(2048, 2048), (2048, 2048)]
    print(f"結果：{'通過' if name_ok and size_ok else '失敗'}")
