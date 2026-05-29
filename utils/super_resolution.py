"""Real-ESRGAN 超解析度模組（RealESRGAN_x4plus_anime_6B）。

x4plus 一次放大 4 倍，以 while 迴圈反覆執行直到短邊 ≥ TARGET_SIZE。
VRAM 管理（3090 24GB）：fp16 + 固定 tile=1024 + 每輪後 empty_cache，
避免 caching allocator 累積殘留導致後輪 OOM。失敗時 fallback cv2.resize。
"""
from __future__ import annotations
import sys
import warnings
from pathlib import Path
import cv2
import numpy as np

_THIS_DIR = Path(__file__).parent
WEIGHTS_PATH = _THIS_DIR.parent / "weights" / "RealESRGAN_x4plus_anime_6B.pth"

SR_SCALE = 4
TARGET_SIZE = 4096
TILE_SIZE = 1024
TILE_PAD = 10

_upsampler = None


def _load_model():
    """延遲載入 Real-ESRGAN 模型並 cache，後續多張臉共用。"""
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

        # 6B = 6 個 residual blocks，其餘為 x4plus_anime_6B 固定架構參數
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,
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
            half=True,
            gpu_id=0 if use_cuda else None,
        )

    return _upsampler


def upscale_image(image: np.ndarray) -> np.ndarray:
    """用 Real-ESRGAN 將 RGB 圖片放大至短邊 ≥ TARGET_SIZE。

    失敗時 fallback cv2.resize（INTER_CUBIC）一次性放大到所需倍率並印警告。
    """
    try:
        import torch
        upsampler = _load_model()
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        pass_index = 0
        while min(bgr.shape[:2]) < TARGET_SIZE:
            pass_index += 1
            in_h, in_w = bgr.shape[:2]
            upsampler.tile_size = TILE_SIZE
            print(f"SR pass #{pass_index} 輸入 {in_w}×{in_h}、tile={TILE_SIZE}", flush=True)

            bgr, _ = upsampler.enhance(bgr, outscale=SR_SCALE)

            out_h, out_w = bgr.shape[:2]
            print(f"SR pass #{pass_index} 輸出 {out_w}×{out_h}", flush=True)

            # 釋放 caching allocator 殘留，避免多輪累積導致後輪 OOM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    except Exception as exc:
        print(
            f"警告：Real-ESRGAN 執行失敗（{exc}），改用 cv2.resize 放大至短邊 ≥ {TARGET_SIZE}。",
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

    # 縮成 800×800 模擬裁切後 < 4096，會跑 2 次 SR（800 → 3200 → 12800）
    small = cv2.resize(img, (800, 800), interpolation=cv2.INTER_AREA)
    print(f"輸入（縮小後）：{small.shape}")
    print(f"目標短邊 ≥ {TARGET_SIZE}")
    print("執行 Real-ESRGAN，首次需載入模型請稍候")
    upscaled = upscale_image(small)
    print(f"放大後：{upscaled.shape}")

    out_dir = Path(__file__).parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    Image.fromarray(upscaled).save(out_dir / "_sr_test.png")
    print(f"已輸出測試圖：{out_dir / '_sr_test.png'}")
