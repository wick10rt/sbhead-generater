"""Real-ESRGAN 超解析度模組。

使用 RealESRGAN_x4plus_anime_6B 模型對動漫圖片做超解析度放大。
x4plus 一次只能放大 4 倍，為達到目標尺寸（4096）會以 while 迴圈反覆執行
直到短邊 ≥ TARGET_SIZE 才回傳。

品質設定（3090 24GB 解鎖）：
- fp32（half=False）：動漫平塗區色塊更乾淨
- tile=0 不分塊：邊緣零拼接縫
- 安全機制：每輪 SR 前檢查輸入邊長，> TILE_SAFE_THRESHOLD 時改用
  tile=TILE_SAFE 防 OOM（4096 大圖只切 4 塊、拼接縫肉眼幾乎無感）

模型載入失敗或推論失敗時，自動 fallback 到 cv2.resize（INTER_CUBIC）並印警告。
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
import cv2
import numpy as np

# === Real-ESRGAN 設定 ===
# 權重檔路徑（相對 main.py 同層的 weights/）
_THIS_DIR = Path(__file__).parent
WEIGHTS_PATH = _THIS_DIR.parent / "weights" / "RealESRGAN_x4plus_anime_6B.pth"

# 模型固定 ×4 倍率
SR_SCALE = 4

# 目標輸出短邊（while 迴圈停止條件）
TARGET_SIZE = 4096

# tile 安全機制：輸入邊長 > TILE_SAFE_THRESHOLD 時改用 TILE_SAFE 防 OOM
TILE_SAFE_THRESHOLD = 2048
TILE_SAFE = 2048
TILE_PAD = 10

# 模型物件 cache（首次推論時載入，後續多張臉共用）
_upsampler = None


def _load_model():
    """延遲載入 Real-ESRGAN 模型並 cache。

    建構時用 tile=0；每次推論前由 upscale_image 動態切換 tile。

    Raises:
        FileNotFoundError: 找不到權重檔。
    """
    global _upsampler
    if _upsampler is not None:
        return _upsampler

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"找不到權重檔：{WEIGHTS_PATH}\n"
            f"請至 https://github.com/xinntao/Real-ESRGAN/releases "
            f"下載 RealESRGAN_x4plus_anime_6B.pth 放到 weights/ 內。"
        )

    # 抑制 basicsr / realesrgan 啟動時的相容性 warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        # RealESRGAN_x4plus_anime_6B 的固定架構參數
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,         # 6B = 6 個 residual blocks
            num_grow_ch=32,
            scale=SR_SCALE,
        )

        use_cuda = torch.cuda.is_available()
        _upsampler = RealESRGANer(
            scale=SR_SCALE,
            model_path=str(WEIGHTS_PATH),
            model=model,
            tile=0,                    # 初始不分塊；每輪推論前動態調整
            tile_pad=TILE_PAD,
            pre_pad=0,
            half=False,                # 固定 fp32，追求最高品質
            gpu_id=0 if use_cuda else None,
        )

    return _upsampler


def _pick_tile_for_size(side: int) -> int:
    """依輸入邊長決定本輪 SR 使用的 tile size。

    ≤ TILE_SAFE_THRESHOLD：tile=0（不分塊、零拼接縫）
    > TILE_SAFE_THRESHOLD：tile=TILE_SAFE（防 OOM、4 塊幾乎無縫）
    """
    return 0 if side <= TILE_SAFE_THRESHOLD else TILE_SAFE


def upscale_image(image: np.ndarray) -> np.ndarray:
    """使用 Real-ESRGAN 將 RGB 圖片放大至短邊 ≥ TARGET_SIZE。

    x4plus 每次 ×4，內部 while 迴圈反覆執行直到短邊 ≥ TARGET_SIZE 才回傳。
    每輪 SR 前依輸入邊長動態決定 tile：≤ 2048 用 0、> 2048 用 2048。

    若 SR 失敗（權重缺、CUDA OOM 等），自動 fallback 到 cv2.resize 並印警告，
    fallback 一次性放大到「短邊 ≥ TARGET_SIZE」所需倍率。

    Args:
        image: RGB 格式 numpy 陣列，形狀 (H, W, 3)。

    Returns:
        RGB 圖片，短邊 ≥ TARGET_SIZE。後續由 avatar_output 統一 resize 為
        TARGET_SIZE × TARGET_SIZE 輸出。
    """
    try:
        upsampler = _load_model()
        # RealESRGANer.enhance 內部以 BGR 處理（與 cv2 慣例一致）
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        pass_index = 0
        while min(bgr.shape[:2]) < TARGET_SIZE:
            pass_index += 1
            in_h, in_w = bgr.shape[:2]
            tile = _pick_tile_for_size(max(in_h, in_w))
            upsampler.tile_size = tile   # RealESRGANer 內部用 tile_size 屬性

            tile_label = "0（不分塊）" if tile == 0 else str(tile)
            print(
                f"        SR pass #{pass_index}：輸入 {in_w}×{in_h}、tile={tile_label}",
                flush=True,
            )

            bgr, _ = upsampler.enhance(bgr, outscale=SR_SCALE)

            out_h, out_w = bgr.shape[:2]
            print(
                f"        SR pass #{pass_index}：輸出 {out_w}×{out_h}",
                flush=True,
            )

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    except Exception as exc:
        print(
            f"警告：Real-ESRGAN 執行失敗（{exc}）\n"
            f"      改用 cv2.resize（INTER_CUBIC）一次性放大至短邊 ≥ {TARGET_SIZE}。",
            file=sys.stderr,
        )
        h, w = image.shape[:2]
        scale = TARGET_SIZE / min(h, w)
        return cv2.resize(
            image,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_CUBIC,
        )


if __name__ == "__main__":
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

    # 縮成 800×800 模擬「裁切後 < 4096」的情境，會跑 2 次 SR
    # （800 → 3200 → 12800），第 2 次自動降為 tile=2048
    small = cv2.resize(img, (800, 800), interpolation=cv2.INTER_AREA)
    print(f"輸入（縮小後）：{small.shape}")
    print(f"目標短邊：≥ {TARGET_SIZE}")
    print("執行 Real-ESRGAN（首次可能要載入模型，請稍候）...")
    upscaled = upscale_image(small)
    print(f"放大後：{upscaled.shape}")

    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    Image.fromarray(upscaled).save(out_dir / "_sr_test.png")
    print(f"已輸出測試圖：{out_dir / '_sr_test.png'}")
