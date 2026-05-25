"""大頭貼輸出模組（raw / sr 雙版本同步）。

每張臉同時輸出兩個 1024×1024 PNG 版本：
- `outputs_dir/raw/output(N).png`：未做超解析度的版本
- `outputs_dir/sr/output(N).png`：做完超解析度的版本

兩個子資料夾用同一個編號，方便對比 AI 上採前後的畫質差異。
命名規則：index=0 → `output.png`、index=N → `output(N).png`。
編號採「raw/ 與 sr/ 兩邊現有最大編號 + 1」，不填補編號空缺。
"""
from __future__ import annotations
import re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

OUTPUT_SIZE = 1024
RAW_SUBDIR = "raw"
SR_SUBDIR = "sr"

# 比對 output.png 或 output(N).png
_FILENAME_PATTERN = re.compile(r"^output(?:\((\d+)\))?\.png$")


def _index_to_filename(index: int) -> str:
    """index=0 → output.png；index=N → output(N).png。"""
    if index == 0:
        return "output.png"
    return f"output({index}).png"


def _max_index_in_dir(folder: Path) -> int:
    """掃描資料夾，回傳現有最大編號。沒有任何符合檔案則回傳 -1。

    output.png 視為編號 0；output(N).png 視為編號 N。
    """
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
    """同時掃 raw/ 與 sr/，回傳本次執行可用的起始 index。

    取兩邊最大編號 + 1，確保 raw 與 sr 編號永遠同步、不會覆寫舊檔。
    若兩邊都沒有任何 output*.png，回傳 0（也就是 output.png）。

    Args:
        outputs_dir: 輸出根資料夾（含 raw/ 與 sr/）。

    Returns:
        下一張臉應使用的 index。
    """
    raw_dir = outputs_dir / RAW_SUBDIR
    sr_dir = outputs_dir / SR_SUBDIR
    max_num = max(_max_index_in_dir(raw_dir), _max_index_in_dir(sr_dir))
    return max_num + 1


def _resize_to_output(image: np.ndarray) -> np.ndarray:
    """將圖片 resize 到 OUTPUT_SIZE × OUTPUT_SIZE。

    縮小用 INTER_AREA、放大用 INTER_CUBIC；已是目標尺寸則直接回傳。
    """
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
    """同時將 raw 與 sr 兩張圖輸出為 1024×1024 PNG。

    raw 寫到 `outputs_dir/raw/`、sr 寫到 `outputs_dir/sr/`，
    兩邊使用同一個 index 對應的檔名（保證對比時編號一致）。

    使用 PIL 寫檔，避免 cv2.imwrite 在 Windows 中文路徑下失敗。

    Args:
        raw_image: 未做 SR 的 RGB 圖片。
        sr_image: 做完 SR 的 RGB 圖片（若裁切 ≥ 1024 可能與 raw 相同）。
        outputs_dir: 輸出根資料夾（會自動建立 raw/ 與 sr/）。
        index: 本張臉的編號（由 next_paired_index 取得後逐張 +1）。

    Returns:
        (raw_path, sr_path)：實際輸出的兩個檔案路徑。
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
    # 測試 raw/sr 同步編號邏輯
    test_dir = Path(__file__).parent.parent / "outputs" / "_test_paired_naming"
    # 清空測試資料夾
    if test_dir.exists():
        for sub in (test_dir / RAW_SUBDIR, test_dir / SR_SUBDIR):
            if sub.exists():
                for f in sub.iterdir():
                    f.unlink()
                sub.rmdir()
        test_dir.rmdir()

    fake_raw = np.zeros((512, 512, 3), dtype=np.uint8)
    fake_sr = np.full((2048, 2048, 3), 200, dtype=np.uint8)

    # 模擬連跑三張臉
    results = []
    for _ in range(3):
        idx = next_paired_index(test_dir)
        raw_p, sr_p = save_paired_avatar(fake_raw, fake_sr, test_dir, idx)
        results.append((raw_p.name, sr_p.name))

    print("連續輸出三張臉的檔名測試：")
    for i, (r, s) in enumerate(results, start=1):
        print(f"  第 {i} 張：raw={r} / sr={s}")

    expected = [
        ("output.png", "output.png"),
        ("output(1).png", "output(1).png"),
        ("output(2).png", "output(2).png"),
    ]
    print(f"\n預期：{expected}")
    print(f"實際：{results}")
    print(f"結果：{'通過' if results == expected else '失敗'}")
