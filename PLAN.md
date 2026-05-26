# sbhead-generater — 開發計畫與進度追蹤

> 這份計畫由使用者（張胖胖）制定，Claude 必須**嚴格遵循**，不得擅自調整順序或跳階段。
> 每次 Claude 動工後，必須回報「對應到計畫的哪一步、做了什麼、目前進度」。
> Claude 可視情況微調執行順序，以能成功運行為第一準則。

---

## 最終設計決策（不可更改）

| 項目 | 決定 |
|------|------|
| 介面 | `python main.py -i <圖片路徑>`，無其他 flag |
| 輸出位置 | `main.py` 同層 `outputs/raw/` 與 `outputs/sr/` 兩個子資料夾，自動建立 |
| 輸出格式 | 固定 PNG，4096×4096 |
| 輸出命名 | `output.png`、`output(1).png`、`output(2).png`…，raw/ 與 sr/ 兩邊編號同步 |
| 輸入格式 | JPG / JPEG / PNG |
| 臉部偵測 | dghs-imgutils[gpu]（`pip install dghs-imgutils[gpu]`） |
| 多臉策略 | 偵測到的所有臉部全部處理（按偵測器原順序），單張失敗印警告後跳過、繼續其他臉 |
| 偵測失敗 | 無臉時 `sys.exit(1)` + 清楚錯誤訊息，不輸出任何檔案 |
| 裁切外擴倍率 | EXPAND_RATIO=1.6（bbox 寬高最大邊 ×1.6），EXTRA_TOP_RATIO=0.15 |
| Enhance | 永遠自動執行（銳化 + 降噪 + 對比），raw 與 sr 兩版都套 |
| 雙版本輸出 | 每張臉同時輸出 raw 版（不跑 SR）與 sr 版（跑 SR）以便比對；兩版都 4096×4096 |
| SR 條件 | 裁切後尺寸 < 4096 → sr 版**連續跑 Real-ESRGAN 直到 ≥ 4096**；≥ 4096 → sr 版與 raw 版內容相同（都直接 resize） |
| SR 模型 | `RealESRGAN_x4plus_anime_6B.pth`，放 `weights/`；tile=0 不分塊（輸入 > 2048 自動降為 tile=2048 防 OOM）；fp32（3090 24GB 解鎖最高品質設定） |
| 路徑處理 | pathlib + PIL |
| 中間結果 | 不儲存 |
| 平台 | Windows 11 + NVIDIA RTX 3090 24GB |
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
    ├── raw/                              # 未做 SR 的版本
    └── sr/                               # 做完 SR 的版本（檔名與 raw/ 同步）
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

- [x] 2.1 `utils/avatar_output.py` — `save_avatar(image, outputs_dir)`
  - 固定輸出 1024×1024 PNG
  - 命名規則：`output.png` → `output(1).png` → `output(2).png`…
  - 自動建立 `outputs/` 資料夾

- [x] 2.2 `utils/enhance.py` — `enhance_image(image)`
  - 銳化（unsharp mask）
  - 降噪（bilateral filter）
  - 對比度調整（CLAHE）
  - 保守參數，避免動漫線條變奇怪

- [x] 2.3 `utils/crop_avatar.py` — `crop_by_bbox(image, bbox)`
  - 依臉部 bbox 擴大裁切範圍（含頭髮與部分肩膀）
  - 上方擴展比下方多，符合動漫頭部比例
  - 裁成正方形，修正超出圖片邊界

- [x] 2.4 `utils/face_detect.py` — `detect_largest_face(image) -> bbox`
  - 使用 dghs-imgutils 偵測動漫臉部
  - 回傳面積最大的臉部 bbox（x0, y0, x1, y1）
  - 無臉偵測到 → `sys.exit(1)` + 清楚錯誤訊息

- [x] 2.5 `utils/super_resolution.py` — `upscale_image(image)`
  - 載入 `RealESRGAN_x4plus_anime_6B`（路徑相對 main.py）
  - 優先使用 CUDA GPU
  - 執行失敗時 fallback 到 `cv2.resize` 並印警告

- [x] 2.6 **驗證點**：每個模組的 `if __name__ == "__main__":` 區塊單獨測試通過

---

## 階段 3：整合 main.py

- [x] 3.1 argparse：只有 `-i / --input`（圖片路徑）
- [x] 3.2 流程串接：
  1. 驗證輸入檔案存在且格式正確（JPG/JPEG/PNG）
  2. pathlib + PIL 讀圖
  3. 臉部偵測（失敗 → sys.exit(1)）
  4. 依 bbox 裁切頭像
  5. enhance（永遠執行）
  6. 判斷裁切後尺寸：< 1024 → SR；≥ 1024 → 直接 resize
  7. Resize 至 1024×1024
  8. 存到 `outputs/`（自動建立資料夾）
- [x] 3.3 每一步加 print 訊息，讓終端機看得到進度
- [x] 3.4 錯誤訊息清楚指出是哪一步失敗

---

## 階段 4：端到端測試

- [x] 4.1 基本跑通：`python main.py -i sample_images/xxx.jpg`
- [x] 4.2 確認輸出是 1024×1024 PNG 且存在 `outputs/`
- [x] 4.3 連跑兩次同一張圖，確認 `output(1).png` 自動命名正確
- [x] 4.4 故意餵沒有臉的圖，確認 `sys.exit(1)` 錯誤訊息清楚
- [x] 4.5 故意餵不存在的路徑，確認錯誤訊息清楚
- [x] 4.6 故意餵高解析度大圖（裁切後 ≥ 1024），確認跳過 SR 只做 resize
- [x] 4.7 故意移除權重檔，確認 SR fallback 到 cv2.resize 並印警告
- [x] 4.8 確認 GPU 正確使用（`nvidia-smi` 確認顯卡運行中）

---

## 階段 5：收尾文件

- [x] 5.1 改寫 `README.md`（安裝步驟、權重下載、執行方式、basicsr 修補說明）
- [x] 5.2 確認 `計畫書.md` 反映最終實作
- [x] 5.3 `git add . && git commit && git push`

---

## 階段 6：v2 改版（多臉處理 + raw/sr 雙版本對比）

> 第一版完成、本機測試通過後追加的需求。

**變更摘要：**
- crop 外擴倍率 1.9 → 1.6（EXTRA_TOP_RATIO 維持 0.15）
- 多臉處理：偵測到的所有臉部全部處理（按偵測器原順序），單張失敗跳過繼續
- 雙版本輸出：raw（未跑 SR）與 sr（跑 SR）兩個版本，分別存到 `outputs/raw/` 與 `outputs/sr/`
- 命名規則：raw 與 sr 編號同步；同一次執行內多張臉用連續編號

### 6.0 規格文件同步（先動 MD，再動程式碼）

- [x] 6.0.1 更新 `CLAUDE.md`：設計決策表、資料夾結構、處理流程、模組規格
- [x] 6.0.2 更新 `PLAN.md`：本變更紀錄與階段 6 任務清單
- [x] 6.0.3 更新 `README.md`：整合計畫書內容 + 執行方式、輸出位置說明、專案結構樹
- [x] 6.0.4 刪除 `計畫書.md`（內容已整合進 README.md）

### 6.1 程式碼修改

- [x] 6.1.1 `utils/crop_avatar.py`：`EXPAND_RATIO` 1.9 → 1.6（分兩次 commit 調整）
- [x] 6.1.2 `utils/face_detect.py`：
  - `detect_largest_face` → `detect_all_faces`，回傳所有 bbox（依偵測器原順序）
- [x] 6.1.3 `utils/avatar_output.py`：
  - 新增 `next_paired_index(outputs_dir)`：掃 raw/ 與 sr/ 取兩邊最大編號 + 1
  - 新增 `save_paired_avatar(raw_image, sr_image, outputs_dir, index)`：同時寫入 raw/ 與 sr/ 同名檔
  - 移除舊的 `save_avatar` 與 `_next_output_path`
- [x] 6.1.4 `utils/__init__.py`：更新 re-export
- [x] 6.1.5 `main.py`：
  - 改用 `detect_all_faces`
  - 先呼叫 `next_paired_index` 取得起始 index
  - 對每張臉做 try/except，單張失敗印警告後 continue
  - 進度訊息改成兩段式：「[N/M] 第 X 張臉 – 裁切...」
  - 結尾印出本次成功輸出的所有檔案路徑與成功 X/N 張

### 6.2 端到端測試

- [x] 6.2.1 單臉圖：確認 `outputs/raw/output.png` 與 `outputs/sr/output.png` 同步產生
- [x] 6.2.2 多臉圖：確認 N 張臉產生 N 對檔案、編號連續、raw/sr 兩邊同步
- [x] 6.2.3 連跑兩次：確認編號接續、不覆蓋既有檔案
- [x] 6.2.4 1.6x 視覺確認：對比舊版本與 1.6x 的裁切結果
- [x] 6.2.5 大圖（裁切 ≥ 1024）：確認 raw 與 sr 兩版內容相同（都跳過 SR）
- [x] 6.2.6 無臉圖：`sys.exit(1)` 行為不變
- [x] 6.2.7 模型 cache：多臉時確認 Real-ESRGAN 只載入一次

### 6.3 收尾

- [x] 6.3.1 `git add . && git commit && git push`

---

## 階段 7：v3 改版（輸出 4096×4096）

> v2 完成後追加的需求：把輸出解析度從 1024×1024 拉到 4096×4096，
> 用以凸顯 Real-ESRGAN 對「低解析度上採」的優勢。

**變更摘要：**
- `OUTPUT_SIZE` 全面 1024 → 4096
- `upscale_image` 改為迴圈：x4plus 一次 ×4，反覆執行直到輸出邊長 ≥ 4096
- raw 版維持「不跑 SR、直接傳統 resize 到 4096」（明顯模糊正是要顯示 SR 的優勢）
- 觸發條件：裁切 < 4096 跑 SR；裁切 ≥ 4096 跳過（sr = raw）
- 不去背（評估後使用者決定不做）
- **GPU 從 4060 8GB 升級為 3090 24GB**，連帶解鎖兩項最高品質設定：
  - `tile=400` → `tile=0`（不分塊）：徹底消除 tile 拼接縫；但因 24GB 仍可能在「小裁切連跑 2 次 SR」的第 2 次（輸入可達 3200+）OOM，加入安全機制：**SR 前檢查輸入邊長，> 2048 自動降為 tile=2048**（4096 大圖只切 4 塊、拼接縫幾乎看不見）
  - `half=True (fp16)` → `half=False (fp32)`：動漫平塗區數值精度更高，色塊更乾淨

**已知副作用：**
- 單張 PNG ≈ 5–20 MB，多臉合照單次執行可能輸出 50–200 MB
- 小裁切（< 1024）需跑 2 次 SR pass，但因 tile=0/2048、fp32 一次推更多像素，總時間視 GPU 利用率而定
- fp32 約 2× VRAM、2× 計算量；3090 24GB 仍十分充裕

### 7.0 規格文件同步（先動 MD，再動程式碼）

- [x] 7.0.1 更新 `CLAUDE.md`：設計決策表、處理流程、模組規格
- [x] 7.0.2 更新 `PLAN.md`：本變更紀錄與階段 7 任務清單
- [x] 7.0.3 更新 `README.md`：所有 1024 改為 4096、處理流程章節同步

### 7.1 程式碼修改

- [x] 7.1.1 `main.py`：`OUTPUT_SIZE` 1024 → 4096
- [x] 7.1.2 `utils/avatar_output.py`：`OUTPUT_SIZE` 1024 → 4096
- [x] 7.1.3 `utils/super_resolution.py`：
  - 新增 `TARGET_SIZE = 4096`、`TILE_SAFE_THRESHOLD = 2048`、`TILE_SAFE = 2048` 常數
  - `RealESRGANer(half=False)` 永遠 fp32（不再依 CUDA 切換）
  - 改為「每次 SR pass 動態決定 tile size」：輸入邊長 ≤ TILE_SAFE_THRESHOLD 用 tile=0（不分塊），> TILE_SAFE_THRESHOLD 用 tile=TILE_SAFE（=2048）防 OOM
    - 實作上以 `upsampler.tile_size = tile` 動態修改屬性，毋須重建 upsampler，模型權重只載入一次
  - `upscale_image` 改為 while 迴圈，反覆 SR 直到 min(h, w) ≥ TARGET_SIZE
  - 印出每一輪 SR 的輸入 / 輸出尺寸與本輪使用的 tile，方便 demo 觀察
  - fallback（cv2.resize）改為一次性放大到「目標 ÷ 短邊」倍率

### 7.2 端到端測試

- [ ] 7.2.1 小裁切（< 1024）：確認連續跑 2 次 SR、最終輸出 4096
- [ ] 7.2.2 中裁切（1024 ≤ 裁切 < 4096）：確認跑 1 次 SR + resize 4096
- [ ] 7.2.3 大裁切（≥ 4096）：確認跳過 SR、raw 與 sr 內容相同
- [ ] 7.2.4 多臉圖：確認模型仍只載入一次、多次 SR 不會重新載入
- [ ] 7.2.5 3090 24GB VRAM：
  - 確認 tile=0 在「輸入 ≤ 2048」情境下不會 OOM
  - 確認 tile=2048 在「輸入 3200~3500」情境下不會 OOM
  - 用 nvidia-smi 觀察峰值 VRAM；若意外 OOM 確認 fallback 正常觸發
- [ ] 7.2.6 檔案大小檢查：確認 PNG 在合理範圍（< 30 MB）

### 7.3 收尾

- [x] 7.3.1 `git add . && git commit && git push`

---

## 變更紀錄（Changelog）

- 2026-05-15｜（計畫建立）｜由使用者口述，Claude 寫入此檔案作為進度追蹤基準。
- 2026-05-16｜1.1｜建立 `requirements.txt`，鎖定 numpy<2.0、torchvision<0.18。
- 2026-05-16｜1.2｜建立 `.gitignore`。
- 2026-05-16｜1.3（規範更新）｜使用者指定使用 conda 作為虛擬環境管理工具。
- 2026-05-24｜全面重新設計｜根據使用者確認，更新設計決策：移除所有 CLI flag（只保留 -i）、移除 intermediate/ 資料夾、改用 dghs-imgutils 取代 anime-face-detector、輸出固定 1024×1024 PNG、SR 依裁切後尺寸自動決定是否執行、臉部偵測失敗直接 sys.exit(1)、輸出命名採自動編號。
- 2026-05-24｜階段 1 完成｜使用者本機完成 conda 環境、套件安裝、權重下載、basicsr 手改、測試圖放置、驗證指令通過。階段 1 全部勾選完成，可進入階段 2 程式碼開發。
- 2026-05-25｜階段 2～5.2 完成｜五個 utils 模組全部實作完成，main.py 整合完成，端到端測試全數通過（使用者本機驗證）。README.md 與計畫書.md 已同步最終設計。剩餘 5.3 最終 commit & push。
- 2026-05-25｜階段 5.3 完成、進入 v2 改版｜v1 push 完成。使用者提出 3 項變更：(1) crop EXPAND_RATIO 1.9 → 1.6（分兩次調整）；(2) 多臉全處理（按偵測器原順序、單張失敗跳過）；(3) raw/sr 雙版本輸出至 `outputs/raw/` 與 `outputs/sr/`（兩邊編號同步、enhance 兩版都套、≥1024 時兩版相同）。新增階段 6。
- 2026-05-25｜階段 6 完成｜v2 程式碼（多臉 + 雙版本）push 完成。README.md 整合計畫書內容後，計畫書.md 刪除。EXPAND_RATIO 最終定為 1.6。所有 MD 文件內容一致。
- 2026-05-26｜進入 v3 改版｜使用者要求輸出解析度 1024 → 4096。原本同時想加去背功能，評估後（動漫去背需 anime 專用 ISNet 模型、檔案大小翻倍、效果視輸入而定）使用者決定不做。新增階段 7：OUTPUT_SIZE 全面 1024 → 4096、`upscale_image` 改為 while 迴圈反覆 SR 直到 ≥ 4096、raw 也拉到 4096 以凸顯 SR 優勢。
- 2026-05-26｜v3 規格追加（GPU 升級）｜使用者本機從 RTX 4060 8GB 升級至 RTX 3090 24GB，連帶調整最高品質設定：`tile=400` → `tile=1024`（消除 4096 大圖的 tile 拼接縫）、`half=True (fp16)` → `half=False (fp32)`（提升動漫平塗區色塊純度）。輸出解析度維持 4096、enhance 參數維持保守不動、SR 模型仍用 6B anime 專用版（換 23B 通用版反而會破壞動漫風）。
- 2026-05-26｜v3 規格再追加（tile=0）｜使用者要求 tile 改為 0（不分塊，零拼接縫）。經估算：小裁切連跑 2 次 SR 的第 2 次（輸入可達 3200+）+ fp32 + tile=0 預估 25–35GB VRAM、會 OOM。決議：每次 SR pass 前動態決定 tile size，**輸入 ≤ 2048 用 tile=0、> 2048 用 tile=2048**。實務上 4096 大圖只切 4 塊、拼接縫肉眼幾乎無感。Enhance amount 維持 0.3（評估後使用者選擇不加重，避免 SR 後出現 halo）。
- 2026-05-26｜v3 程式碼完成｜階段 7.1 三支程式碼修改完成並 push（commit `b0d5afa`）：`main.py` 與 `utils/avatar_output.py` 的 `OUTPUT_SIZE` 1024 → 4096、`utils/super_resolution.py` 新增 while 迴圈 + 動態 tile 機制（以 `upsampler.tile_size = tile` 動態修改屬性、模型只載入一次）+ fp32 固定。stage 7.2 端到端測試由使用者本機驗證中。
- 2026-05-26｜MD 一致性整理｜CLAUDE.md `save_paired_avatar` 規格描述修正 1024×1024 → 4096×4096（舊版遺漏未改）；README.md 注意 3 措辭調整（移除已不存在的「1024 版本」對照）；PLAN.md 階段 6.2 全數標 [x]（v2 測試使用者已驗證通過）、階段 7.0.3、7.1.x、7.3.1 標 [x] 反映實際進度。歷史性的任務描述（階段 2 的 `save_avatar`、階段 6 的 `detect_largest_face`）刻意保留以呈現演進軌跡，由 Changelog 明確標示時間線。
