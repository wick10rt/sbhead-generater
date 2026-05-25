# sbhead-generater — 開發計畫與進度追蹤

> 這份計畫由使用者（張胖胖）制定，Claude 必須**嚴格遵循**，不得擅自調整順序或跳階段。
> 每次 Claude 動工後，必須回報「對應到計畫的哪一步、做了什麼、目前進度」。
> Claude 可視情況微調執行順序，以能成功運行為第一準則。

---

## 最終設計決策（不可更改）

| 項目 | 決定 |
|------|------|
| 介面 | `python main.py -i <圖片路徑>`，無其他 flag |
| 輸出位置 | `main.py` 同層 `outputs/` 資料夾，自動建立 |
| 輸出格式 | 固定 PNG，1024×1024 |
| 輸出命名 | `output.png`、`output(1).png`、`output(2).png`… |
| 輸入格式 | JPG / JPEG / PNG |
| 臉部偵測 | dghs-imgutils（`pip install imgutils`） |
| 偵測失敗 | `sys.exit(1)` + 清楚錯誤訊息，不輸出任何檔案 |
| Enhance | 永遠自動執行（銳化 + 降噪 + 對比） |
| SR 條件 | 裁切後尺寸 < 1024 → 跑 Real-ESRGAN；≥ 1024 → 直接 resize |
| SR 模型 | `RealESRGAN_x4plus_anime_6B.pth`，放 `weights/` |
| 路徑處理 | pathlib + PIL |
| 中間結果 | 不儲存 |
| 平台 | Windows 11 + NVIDIA 4060+ |
| 程式碼註解 | 繁體中文 |
| basicsr 修補 | 手改套件原始碼（見 1.5） |

---

## 進度標記說明

- `[ ]` 未開始
- `[~]` 進行中
- `[x]` 已完成
- `[!]` 卡住 / 需使用者介入

---

## 資料夾結構（最終版）

```
sbhead-generater/
├── main.py
├── requirements.txt
├── README.md
├── PLAN.md
├── 計畫書.md
├── utils/
│   ├── __init__.py
│   ├── face_detect.py
│   ├── crop_avatar.py
│   ├── enhance.py
│   ├── super_resolution.py
│   └── avatar_output.py
├── weights/
│   └── RealESRGAN_x4plus_anime_6B.pth  ← gitignored，需自行下載
├── sample_images/                        ← gitignored，放測試圖
└── outputs/                              ← gitignored，main.py 自動建立
```

---

## 階段 1：環境準備

> 先處理裝環境，因為這是最容易卡住的地方。

- [x] 1.1 寫 `requirements.txt`（OpenCV、Pillow、NumPy、PyTorch、realesrgan、basicsr、imgutils）
- [x] 1.2 寫 `.gitignore`
- [x] 1.3 建 conda 虛擬環境並安裝套件
  ```bash
  conda create -n sbhead python=3.10 -y
  conda activate sbhead
  # 依顯卡安裝 PyTorch（NVIDIA）：
  conda install pytorch=2.2.2 torchvision=0.17.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
  pip install -r requirements.txt
  ```
- [x] 1.4 下載 `RealESRGAN_x4plus_anime_6B.pth` 放 `weights/`
  - 下載來源：https://github.com/xinntao/Real-ESRGAN/releases
- [x] 1.5 手改 basicsr 套件原始碼
  ```
  檔案位置：<conda env>/Lib/site-packages/basicsr/data/degradations.py
  將：from torchvision.transforms.functional_tensor import rgb_to_grayscale
  改為：from torchvision.transforms.functional import rgb_to_grayscale
  ```
- [x] 1.6 放 1～2 張測試圖到 `sample_images/`
- [x] 1.7 **驗證點**：`python -c "import cv2, torch, realesrgan; from imgutils.detect import detect_faces; print('OK')"` 不報錯

---

## 階段 2：建 utils 五個模組

> 每個模組都能單獨 import 測試，不依賴 main.py。

- [ ] 2.1 `utils/avatar_output.py` — `save_avatar(image, outputs_dir)`
  - 固定輸出 1024×1024 PNG
  - 命名規則：`output.png` → `output(1).png` → `output(2).png`…
  - 自動建立 `outputs/` 資料夾

- [ ] 2.2 `utils/enhance.py` — `enhance_image(image)`
  - 銳化（unsharp mask）
  - 降噪（bilateral filter）
  - 對比度調整（CLAHE）
  - 保守參數，避免動漫線條變奇怪

- [ ] 2.3 `utils/crop_avatar.py` — `crop_by_bbox(image, bbox)`
  - 依臉部 bbox 擴大裁切範圍（含頭髮與部分肩膀）
  - 上方擴展比下方多，符合動漫頭部比例
  - 裁成正方形，修正超出圖片邊界

- [ ] 2.4 `utils/face_detect.py` — `detect_largest_face(image) -> bbox`
  - 使用 dghs-imgutils 偵測動漫臉部
  - 回傳面積最大的臉部 bbox（x0, y0, x1, y1）
  - 無臉偵測到 → `sys.exit(1)` + 清楚錯誤訊息

- [ ] 2.5 `utils/super_resolution.py` — `upscale_image(image)`
  - 載入 `RealESRGAN_x4plus_anime_6B`（路徑相對 main.py）
  - 優先使用 CUDA GPU
  - 執行失敗時 fallback 到 `cv2.resize` 並印警告

- [ ] 2.6 **驗證點**：每個模組的 `if __name__ == "__main__":` 區塊單獨測試通過

---

## 階段 3：整合 main.py

- [ ] 3.1 argparse：只有 `-i / --input`（圖片路徑）
- [ ] 3.2 流程串接：
  1. 驗證輸入檔案存在且格式正確（JPG/JPEG/PNG）
  2. pathlib + PIL 讀圖
  3. 臉部偵測（失敗 → sys.exit(1)）
  4. 依 bbox 裁切頭像
  5. enhance（永遠執行）
  6. 判斷裁切後尺寸：< 1024 → SR；≥ 1024 → 直接 resize
  7. Resize 至 1024×1024
  8. 存到 `outputs/`（自動建立資料夾）
- [ ] 3.3 每一步加 print 訊息，讓終端機看得到進度
- [ ] 3.4 錯誤訊息清楚指出是哪一步失敗

---

## 階段 4：端到端測試

- [ ] 4.1 基本跑通：`python main.py -i sample_images/xxx.jpg`
- [ ] 4.2 確認輸出是 1024×1024 PNG 且存在 `outputs/`
- [ ] 4.3 連跑兩次同一張圖，確認 `output(1).png` 自動命名正確
- [ ] 4.4 故意餵沒有臉的圖，確認 `sys.exit(1)` 錯誤訊息清楚
- [ ] 4.5 故意餵不存在的路徑，確認錯誤訊息清楚
- [ ] 4.6 故意餵高解析度大圖（裁切後 ≥ 1024），確認跳過 SR 只做 resize
- [ ] 4.7 故意移除權重檔，確認 SR fallback 到 cv2.resize 並印警告
- [ ] 4.8 確認 GPU 正確使用（`nvidia-smi` 確認顯卡運行中）

---

## 階段 5：收尾文件

- [ ] 5.1 改寫 `README.md`（安裝步驟、權重下載、執行方式、basicsr 修補說明、FAQ）
- [ ] 5.2 確認 `計畫書.md` 反映最終實作
- [ ] 5.3 `git add . && git commit && git push`

---

## 變更紀錄（Changelog）

- 2026-05-15｜（計畫建立）｜由使用者口述，Claude 寫入此檔案作為進度追蹤基準。
- 2026-05-16｜1.1｜建立 `requirements.txt`，鎖定 numpy<2.0、torchvision<0.18。
- 2026-05-16｜1.2｜建立 `.gitignore`。
- 2026-05-16｜1.3（規範更新）｜使用者指定使用 conda 作為虛擬環境管理工具。
- 2026-05-24｜全面重新設計｜根據使用者確認，更新設計決策：移除所有 CLI flag（只保留 -i）、移除 intermediate/ 資料夾、改用 dghs-imgutils 取代 anime-face-detector、輸出固定 1024×1024 PNG、SR 依裁切後尺寸自動決定是否執行、臉部偵測失敗直接 sys.exit(1)、輸出命名採自動編號。
- 2026-05-24｜階段 1 完成｜使用者本機完成 conda 環境、套件安裝、權重下載、basicsr 手改、測試圖放置、驗證指令通過。階段 1 全部勾選完成，可進入階段 2 程式碼開發。
