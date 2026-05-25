# sbhead-generater — Claude 專案脈絡設定

> 這個檔案是給 Claude 讀的專案上下文。
> Clone 此 repo 後，Claude 開啟此資料夾即可載入相同的專案背景。
> 詳細開發進度與決策紀錄請見 `PLAN.md`。

---

## 角色設定

你是一位擅長 Python、影像處理、AI 超解析度與命令列工具開發的專案助教。

本專案主題：**「Bang Dream 角色大頭貼自動生成系統」**

---

## 專案目標

使用者在本機端透過終端機執行 Python 程式，輸入一張動漫角色圖片路徑後，系統自動偵測臉部、裁切頭像、進行畫質增強與 AI 超解析度放大，輸出 1024×1024 PNG 大頭貼。

不需要網頁介面、GUI、Streamlit、Gradio。只需要終端機指令執行。

---

## 最終設計決策

| 項目 | 決定 |
|------|------|
| 執行指令 | `python main.py -i <圖片路徑>`，僅此一個 flag |
| 輸出位置 | `main.py` 同層 `outputs/` 資料夾下分 `raw/` 與 `sr/` 兩個子資料夾 |
| 輸出格式 | 固定 PNG，1024×1024 |
| 輸出命名 | `raw/output.png` + `sr/output.png`；多臉或重跑時 `output(1).png`、`output(2).png`…，raw 與 sr 兩邊編號同步 |
| 輸入格式 | JPG / JPEG / PNG |
| 臉部偵測 | dghs-imgutils[gpu]（`pip install dghs-imgutils[gpu]`） |
| 多臉策略 | 偵測到的所有臉部全部處理（按偵測器原順序），單張失敗印警告後跳過、繼續處理其他臉 |
| 偵測失敗 | 無臉時 `sys.exit(1)` + 清楚錯誤訊息，不輸出任何檔案 |
| 裁切外擴倍率 | EXPAND_RATIO=1.7（bbox 寬高最大邊 ×1.7），EXTRA_TOP_RATIO=0.15（向上偏移以包含頭髮） |
| Enhance | 永遠自動執行（銳化 + 降噪 + 對比），raw 與 sr 兩版都套 |
| 雙版本輸出 | 每張臉同時輸出 raw 版（不跑 SR）與 sr 版（跑 SR）以便比對 |
| SR 條件 | 裁切後尺寸 < 1024 → sr 版跑 Real-ESRGAN；≥ 1024 → sr 版與 raw 版內容相同（都直接 resize） |
| SR 模型 | `RealESRGAN_x4plus_anime_6B.pth`，放 `weights/` |
| 路徑處理 | pathlib + PIL |
| 中間結果 | 不儲存 |
| 平台 | Windows 11 + NVIDIA 4060+ |
| 程式碼註解 | 繁體中文 |
| basicsr 修補 | 手改套件原始碼（見 README.md） |

---

## 使用工具

| 套件 | 用途 |
|------|------|
| Python 3.10 | 主要開發語言 |
| argparse | 建立終端機指令參數（只有 `-i`） |
| OpenCV | 影像裁切、resize、銳化、降噪、對比度調整 |
| Pillow | 圖片讀取、格式轉換、PNG 輸出 |
| NumPy | 處理圖片陣列與像素資料 |
| dghs-imgutils | 偵測動漫角色臉部位置 |
| Real-ESRGAN | AI 超解析度放大（RealESRGAN_x4plus_anime_6B） |
| PyTorch 2.2.2 | 支援 Real-ESRGAN 模型執行 |

---

## 資料夾結構

```
sbhead-generater/
├── main.py                      # 主程式，串接所有流程
├── requirements.txt             # 套件清單
├── README.md                    # 使用者本機設定步驟
├── PLAN.md                      # 開發進度追蹤
├── 計畫書.md                    # 給教授看的計畫書（繁體中文）
├── utils/
│   ├── __init__.py
│   ├── face_detect.py           # dghs-imgutils 臉部偵測
│   ├── crop_avatar.py           # bbox 裁切與正方形化
│   ├── enhance.py               # 銳化、降噪、對比度
│   ├── super_resolution.py      # Real-ESRGAN 超解析度
│   └── avatar_output.py         # PNG 輸出與自動命名
├── weights/
│   └── RealESRGAN_x4plus_anime_6B.pth   ← gitignored，需自行下載
├── sample_images/               ← gitignored，放測試圖
└── outputs/                     ← gitignored，main.py 自動建立
    ├── raw/                     # 未做 SR 的版本
    └── sr/                      # 做完 SR 的版本（檔名與 raw/ 同步）
```

---

## 處理流程

```
python main.py -i <圖片路徑>
       ↓
驗證輸入（路徑存在、格式為 JPG/JPEG/PNG）
       ↓
dghs-imgutils 偵測動漫臉部 → 取得 N 個 bbox（按偵測器原順序）
  → N == 0：sys.exit(1) + 錯誤訊息
  → N ≥ 1：先決定本次執行用的「起始編號」base（raw/ 與 sr/ 取兩邊最大編號 + 1）
       ↓
for i in range(N):                ← 對每張臉跑同樣流程
  ├─ 依 bbox 裁切（EXPAND_RATIO=1.7，EXTRA_TOP_RATIO=0.15，正方形化）
  ├─ enhance（銳化 + 降噪 + 對比）
  ├─ raw 版：直接 resize 1024×1024 → 存 outputs/raw/output(base+i).png
  └─ sr 版：
      ├─ 裁切尺寸 < 1024：Real-ESRGAN 放大 → resize 1024 → 存 outputs/sr/output(base+i).png
      └─ 裁切尺寸 ≥ 1024：直接 resize 1024（與 raw 內容相同）→ 存 outputs/sr/output(base+i).png
  ※ 單張臉處理過程中拋例外 → 印警告後 continue 下一張，不中斷整批
       ↓
終端機列出本次輸出的所有檔案路徑、成功 X/N 張
```

---

## 各模組規格

### `utils/face_detect.py`
- 函式：`detect_all_faces(image: np.ndarray) -> list[tuple[int, int, int, int]]`
- 使用 dghs-imgutils 偵測動漫臉部
- 回傳所有臉部 bbox 列表 `[(x0, y0, x1, y1), ...]`，順序依偵測器原順序
- 無臉偵測到：`sys.exit(1)` + 錯誤訊息

### `utils/crop_avatar.py`
- 函式：`crop_by_bbox(image: np.ndarray, bbox: tuple) -> np.ndarray`
- `EXPAND_RATIO = 1.7`、`EXTRA_TOP_RATIO = 0.15`
- 依 bbox 向外擴展（上方額外往上偏移以包含頭髮）
- 裁成正方形，修正超出邊界

### `utils/enhance.py`
- 函式：`enhance_image(image: np.ndarray) -> np.ndarray`
- 銳化（unsharp mask）、降噪（bilateral filter）、對比度（CLAHE）
- 保守參數，避免動漫線條過度處理

### `utils/super_resolution.py`
- 函式：`upscale_image(image: np.ndarray) -> np.ndarray`
- 載入 `weights/RealESRGAN_x4plus_anime_6B.pth`（路徑相對 main.py）
- 優先使用 CUDA GPU
- 執行失敗 → fallback `cv2.resize` + 印警告
- 模型 cache：第一張臉觸發載入後，後續多張臉共用同一個 upsampler 實例

### `utils/avatar_output.py`
- 函式 1：`next_paired_index(outputs_dir: Path) -> int`
  - 同時掃描 `outputs_dir/raw/` 與 `outputs_dir/sr/`，回傳兩邊現有最大編號 + 1 作為本次執行的起始 index
  - `output.png` 視為 index 0，`output(N).png` 視為 index N
- 函式 2：`save_paired_avatar(raw_image, sr_image, outputs_dir: Path, index: int) -> tuple[Path, Path]`
  - 把 raw 與 sr 兩張圖各 resize 到 1024×1024，存到 `outputs_dir/raw/` 與 `outputs_dir/sr/` 同名檔案
  - 命名：index=0 → `output.png`、index=N → `output(N).png`
  - 自動建立 `raw/` 與 `sr/` 子資料夾

---

## 注意事項

1. 程式碼註解使用繁體中文。
2. 路徑處理一律使用 `pathlib.Path`，讀圖用 PIL 再轉 numpy，避免中文路徑問題。
3. 不使用任何 GUI / Web 介面套件。
4. BGR/RGB 轉換：PIL 讀入為 RGB，OpenCV 操作為 BGR，給 Real-ESRGAN 前確認為 RGB。
5. Bang Dream 圖片請使用自己截圖或個人學習用途圖片，避免版權問題。
6. SR fallback（cv2.resize）屬於穩定性保險，不是功能降級。
7. 每個 utils 模組都要有 `if __name__ == "__main__":` 測試區塊。
8. 每一個處理步驟都要有 `print` 進度訊息，方便 demo 錄影時看到流程。
