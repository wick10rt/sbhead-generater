# sbhead-generater
**Bang Dream 角色大頭貼自動生成系統**

> 🟥 **本分支為 AMD GPU 版本**（Linux + ROCm 6.2，目標顯卡 7800XT / 6700XT 等 8~12GB）。
> NVIDIA / CUDA 版請見 `main` 分支。兩者差異僅在 PyTorch 安裝方式、臉部偵測後端
> 與 SR 的 VRAM 參數（tile=512），核心流程相同。

---

## 一、專案介紹

本專案的主題是「Bang Dream 角色大頭貼自動生成系統」。選擇這個題目的原因，是因為平常喜歡看動漫，也常常想把喜歡的角色截圖拿來當作社群帳號的頭像，但每次都要手動裁切、調整大小、修圖，過程繁瑣，因此希望用這學期學到的影像處理技術，製作一個能自動完成這些步驟的命令列小工具。

整個系統以本機端的命令列工具方式呈現，使用者只需在終端機輸入一行指令並指定圖片路徑，程式便會自動偵測畫面中所有角色的臉部位置、裁切頭像區域、進行畫質增強與 AI 超解析度放大，並針對每一張臉同時輸出「未做超解析度」與「做完超解析度」兩個版本的 2048×2048 像素 PNG 大頭貼，方便比對 AI 放大後的畫質差異（sr 版先以 Real-ESRGAN 超採樣到 4096 再縮小，邊緣更平滑）。整個過程不需要開啟任何網頁或視窗介面，操作單純直覺。

---

## 二、專案目標

本專案的目標是將這學期學到的影像處理觀念實際應用在一個完整的小作品上。從讀取圖片、動漫臉部偵測、自動裁切、銳化降噪，到最後使用 AI 模型進行超解析度放大，希望能讓整個處理流程完整串接，並輸出可以實際使用的動漫角色大頭貼。系統會同時保留「未經 AI 超解析度」與「經過 AI 超解析度」兩個版本，藉此清楚展示傳統 resize 與 AI 上採之間的畫質差異，達到觀察與比較的學習目的。

---

## 三、功能與處理流程

系統使用 OpenCV、Pillow 與 NumPy 進行基本的影像讀取與處理，並透過 dghs-imgutils 函式庫偵測動漫角色的臉部位置，自動裁切出包含頭髮與部分肩膀的頭像範圍（bbox 寬高最大邊外擴 1.6 倍、上方額外往上偏移以容納頭髮）。畫面中若有多張臉，會將所有偵測到的臉部全部處理並逐一輸出；若沒有偵測到任何臉，程式將顯示明確的錯誤訊息並終止執行。

裁切完成後，程式自動進行適度的銳化與降噪，避免動漫線條被過度處理（原本的 CLAHE 局部對比度均衡會把臉頰等柔和漸層區壓暗成塊狀髒污，已移除）。接著針對每張臉產出兩個版本：「raw 版」直接以傳統 resize 縮放至 2048×2048；「sr 版」則根據裁切後尺寸做判斷，若小於 4096 便使用 Real-ESRGAN（採用適合動漫圖片的 RealESRGAN_x4plus_anime_6B 權重）進行 AI 超解析度放大，由於 x4plus 模型一次只能放大 4 倍，若一次仍不足會連續執行多次直到尺寸 ≥ 4096，再以 INTER_AREA 縮小至 2048；若已大於等於 4096 則同樣直接 resize。兩版最終都輸出 2048×2048 PNG，分別存放於 `outputs/raw/` 與 `outputs/sr/`，兩邊檔名同步編號，方便對照 AI 上採前後的畫質差異——尤其是裁切尺寸較小的情況，raw 版被傳統 resize 強行拉大會明顯模糊，恰好凸顯 SR 版的優勢（sr 版先超採樣到 4096 再縮小，縮小本身即天然抗鋸齒、邊緣更平滑）。

**處理流程：**

1. 使用者在終端機執行：`python main.py -i <圖片路徑>`
2. 程式讀取圖片（支援 JPG、JPEG、PNG）
3. dghs-imgutils 偵測動漫臉部，取得畫面上所有臉部的 bbox 座標
4. 對每一張偵測到的臉依序執行下列步驟：
   1. 依 bbox 自動裁切頭像（含頭髮與部分肩膀，外擴 1.6 倍）
   2. 自動進行銳化、降噪
   3. **raw 版**：直接 resize 至 2048×2048
   4. **sr 版**：若裁切後 < 4096px 則連續跑 Real-ESRGAN 超解析度放大直到 ≥ 4096，再 INTER_AREA 縮小至 2048×2048；若 ≥ 4096px 則同 raw 版直接 resize
   5. 將 raw 與 sr 兩個版本分別輸出至 `outputs/raw/` 與 `outputs/sr/`（同名同編號）
5. 終端機顯示每張臉的處理進度，最後列出本次輸出的所有檔案路徑與成功張數

---

## 四、使用工具

| 套件 | 用途 |
|------|------|
| Python 3.10 | 主要開發語言 |
| argparse | 建立終端機指令參數（只有 `-i`） |
| OpenCV | 影像裁切、resize、銳化、降噪、對比度調整 |
| Pillow | 圖片讀取、格式轉換、PNG 輸出 |
| NumPy | 處理圖片陣列與像素資料 |
| dghs-imgutils | 偵測動漫角色臉部位置 |
| Real-ESRGAN | AI 超解析度放大，採用 RealESRGAN_x4plus_anime_6B 動漫專用權重 |
| PyTorch (ROCm 6.2) | 支援 Real-ESRGAN 模型執行；ROCm 下 torch.cuda.* 經 HIP 對應到 AMD GPU |

---

## 五、預計展示方式

最終以錄影方式呈現，準備數張不同的 Bang Dream 角色圖片（包含單人與多人合照），展示自動臉部偵測、裁切以及超解析度放大的效果。程式執行時會在終端機逐步印出每張臉的處理進度，並將 raw 版與 sr 版分別存到 `outputs/raw/` 與 `outputs/sr/`。錄影時會把同編號的兩張圖片並排比對，凸顯傳統 resize 與 AI 超解析度放大之間的細節差異，並展示多臉合照能一次處理多張角色的效果。

---

## 六、環境設定

> 程式碼由 Claude 負責，以下步驟需要在本機完成。

### 步驟 1：建立 conda 虛擬環境並安裝套件

```bash
conda create -n sbhead python=3.10 -y
conda activate sbhead
python -m pip install --upgrade pip
```

安裝 PyTorch（AMD 顯卡 ROCm 版）：

> 本分支為 **AMD GPU + Linux + ROCm 6.2** 版本。PyTorch 須使用 ROCm 專用 wheel，
> 不可使用 PyPI 預設版（那是 CUDA build，AMD 不能用）。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
```

> **6700XT / 部分 RDNA2 卡注意**：若 ROCm 未官方支援該 gfx 架構（如 gfx1031），
> 執行時可能需要設定環境變數覆蓋：
> ```bash
> export HSA_OVERRIDE_GFX_VERSION=10.3.0
> ```
> 7800XT（gfx1101）等 RDNA3 卡通常不需要。

安裝其餘套件：

```bash
pip install -r requirements.txt
```

> 注意：`requirements.txt` 已**不含** torch / torchvision（須如上用 ROCm wheel 另裝），
> 且 `dghs-imgutils` 不裝 `[gpu]` extra（onnxruntime-gpu 為 CUDA 專用），臉部偵測走 CPU。

### 步驟 2：下載 Real-ESRGAN 權重檔

前往以下頁面下載 `RealESRGAN_x4plus_anime_6B.pth`：

> https://github.com/xinntao/Real-ESRGAN/releases

下載後放到專案的 `weights/` 資料夾：

```
sbhead-generater/
└── weights/
    └── RealESRGAN_x4plus_anime_6B.pth   ← 放這裡
```

### 步驟 3：手改 basicsr 套件原始碼

安裝完成後，如果執行時出現以下錯誤：

```
ImportError: cannot import name 'rgb_to_grayscale' from 'torchvision.transforms.functional_tensor'
```

找到以下檔案（路徑依你的 conda 環境而定）：

```
~/anaconda3/envs/sbhead/lib/python3.10/site-packages/basicsr/data/degradations.py
```

> 小技巧：可用以下指令直接定位檔案路徑：
> ```bash
> python -c "import basicsr, os; print(os.path.join(os.path.dirname(basicsr.__file__), 'data', 'degradations.py'))"
> ```

開啟後找到這一行：

```python
from torchvision.transforms.functional_tensor import rgb_to_grayscale
```

改成：

```python
from torchvision.transforms.functional import rgb_to_grayscale
```

存檔即可。

### 步驟 4：放入測試圖片

將 1～2 張 Bang Dream 角色圖片放到 `sample_images/` 資料夾（JPG 或 PNG 均可）。

```
sbhead-generater/
└── sample_images/
    └── kasumi.jpg   ← 放這裡
```

> 注意：`sample_images/` 已加入 `.gitignore`，圖片不會被上傳到 GitHub。

### 步驟 5：驗證環境

所有步驟完成後，執行以下指令確認沒有報錯：

```bash
python -c "import cv2, torch, realesrgan; from imgutils.detect import detect_faces; print('環境驗證通過')"
```

確認 ROCm 是否正確啟用（AMD 版重點）：

```bash
python -c "import torch; print('HIP:', torch.version.hip); print('GPU 可用:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '無 GPU')"
```

`HIP` 有版本號、`GPU 可用: True` 並印出你的 AMD 顯卡名稱，代表 ROCm 正常。
若 `HIP: None`，表示裝到 CUDA 版 PyTorch，請重裝 ROCm wheel。

看到 `環境驗證通過` 就代表環境設定完成，可以開始執行程式。

---

## 七、執行方式

```bash
conda activate sbhead
python main.py -i sample_images/kasumi.jpg
```

### 輸出位置與命名規則

輸出結果會分別存到 `outputs/raw/` 與 `outputs/sr/` 兩個子資料夾，兩個版本都是 2048×2048 PNG：

- `outputs/raw/output.png`：未做超解析度的版本（僅裁切 + enhance + 傳統 resize）
- `outputs/sr/output.png`：做完 Real-ESRGAN 超解析度的版本（先超採樣到 4096 再縮小到 2048）

兩邊檔名一律同步，方便對比 SR 效果。多臉時輸出 N 對檔案（按偵測器原順序）、編號連續；重跑時編號從現有最大值 + 1 開始，不覆蓋既有檔案。

> **注意 1**：若裁切後尺寸 ≥ 4096，sr 版會跳過 Real-ESRGAN、直接 resize，此時 raw 版與 sr 版內容相同。
>
> **注意 2**：2048×2048 PNG 單檔約 2–6 MB，多臉合照單次執行可能輸出數十 MB。請留意磁碟空間。
>
> **注意 3**：裁切尺寸越小，需要的 Real-ESRGAN pass 數越多（例：裁切 800 → 3200 → 12800 共 2 次；裁切 167 → 668 → 2672 → 10688 共 3 次），耗時與 VRAM 用量會比裁切較大的情況明顯增加。本 AMD 版針對 8~12GB 顯卡（如 7800XT / 6700XT）使用 tile=512 + fp16 並在每輪後釋放 GPU 殘留記憶體；若仍因 VRAM 不足失敗，會自動 fallback 到傳統 resize 並印警告。

---

## 八、專案結構

```
sbhead-generater/
├── main.py                          # 主程式
├── requirements.txt                 # 套件清單
├── PLAN.md                          # 開發進度追蹤
├── utils/
│   ├── __init__.py
│   ├── face_detect.py               # 動漫臉部偵測
│   ├── crop_avatar.py               # 頭像裁切
│   ├── enhance.py                   # 影像增強
│   ├── super_resolution.py          # Real-ESRGAN 超解析度
│   └── avatar_output.py             # 輸出與命名
├── weights/
│   └── RealESRGAN_x4plus_anime_6B.pth   ← 需自行下載
├── sample_images/                   ← 放測試圖（不進 git）
└── outputs/                         ← 執行結果（不進 git）
    ├── raw/                         # 未做 SR 的版本（2048）
    └── sr/                          # 做完 SR 的版本（2048，由超採樣 4096 縮小，與 raw/ 同名同步）
```
