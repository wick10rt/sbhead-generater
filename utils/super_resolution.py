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
# 512：AMD 8~12GB 安全分塊（1024 易 OOM）
TILE_SIZE = 512
TILE_PAD = 10

_upsampler = None


def _load_model():
    global _upsampler
    if _upsampler is not None:
        return _upsampler

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"找不到權重檔：{WEIGHTS_PATH}\n"
            f"請至 https://github.com/xinntao/Real-ESRGAN/releases "
            f"下載 RealESRGAN_x4plus_anime_6B.pth 放到 weights/ 內。"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        # 6B：num_block=6（x4plus_anime_6B 固定架構）
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
            scale=SR_SCALE,
        )

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            hip_ver = getattr(torch.version, "hip", None)
            backend = f"AMD ROCm (HIP {hip_ver})" if hip_ver else "CUDA"
            print(f"SR 使用 GPU：{torch.cuda.get_device_name(0)}（{backend}）", flush=True)
        else:
            print("SR 未偵測到可用 GPU，將於放大時 fallback cv2.resize", flush=True)

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
