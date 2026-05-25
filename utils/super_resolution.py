"""Real-ESRGAN 超解析度模組。

使用 RealESRGAN_x4plus_anime_6B 模型對動漫圖片做 4 倍超解析度放大。
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

# 放大倍數（x4plus 模型固定 4 倍）
SR_SCALE = 4

# tile size：4060 8GB VRAM 用 400 安全。tile=0 表示不分塊（最快但吃記憶體）
TILE_SIZE = 400
TILE_PAD = 10

# 模型物件 cache（避免重複載入）
_upsampler = None


def _load_model():
    """延遲載入 Real-ESRGAN 模型並 cache。

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
            tile=TILE_SIZE,
            tile_pad=TILE_PAD,
            pre_pad=0,
            half=use_cuda,                # GPU 用 fp16 加速
            gpu_id=0 if use_cuda else None,
        )

    return _upsampler


def upscale_image(image: np.ndarray) -> np.ndarray:
    """使用 Real-ESRGAN 將 RGB 圖片放大 4 倍。

    若 SR 失敗（權重缺、CUDA OOM 等），自動 fallback 到 cv2.resize 並印警告。

    Args:
        image: RGB 格式 numpy 陣列，形狀 (H, W, 3)。

    Returns:
        放大 4 倍後的 RGB 圖片。
    """
    try:
        upsampler = _load_model()
        # RealESRGANer.enhance 內部以 BGR 處理（與 cv2 慣例一致）
        bgr_input = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bgr_output, _ = upsampler.enhance(bgr_input, outscale=SR_SCALE)
        return cv2.cvtColor(bgr_output, cv2.COLOR_BGR2RGB)

    except Exception as exc:
        print(
            f"警告：Real-ESRGAN 執行失敗（{exc}）\n"
            f"      改用 cv2.resize（INTER_CUBIC）作為 fallback。",
            file=sys.stderr,
        )
        h, w = image.shape[:2]
        return cv2.resize(
            image,
            (w * SR_SCALE, h * SR_SCALE),
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

    # 縮成 256×256 模擬「裁切後 < 1024」的情境
    small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    print(f"輸入（縮小後）：{small.shape}")
    print("執行 Real-ESRGAN（首次可能要載入模型，請稍候）...")
    upscaled = upscale_image(small)
    print(f"放大後：{upscaled.shape}")

    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    Image.fromarray(upscaled).save(out_dir / "_sr_test.png")
    print(f"已輸出測試圖：{out_dir / '_sr_test.png'}")
