# sbhead-generater
**Bang Dream 角色大頭貼自動生成系統**

---

## 一、專案介紹

本專案的主題是「Bang Dream 角色大頭貼自動生成系統」。選擇這個題目的原因，是因為平常喜歡看動漫，也常常想把喜歡的角色截圖拿來當作社群帳號的頭像，但每次都要手動裁切、調整大小、修圖，過程繁瑣，因此希望用這學期學到的影像處理技術，製作一個能自動完成這些步驟的命令列小工具。

整個系統以本機端的命令列工具方式呈現，使用者只需在終端機輸入一行指令並指定圖片路徑，程式便會自動偵測畫面中所有角色的臉部位置、裁切頭像區域、進行畫質增強與 AI 超解析度放大，並針對每一張臉同時輸出「未做超解析度」與「做完超解析度」兩個版本的 1024×1024 像素 PNG 大頭貼，方便比對 AI 放大後的畫質差異。整個過程不需要開啟任何網頁或視窗介面，操作單純直覺。

---

## 二、專案目標

本專案的目標是將這學期學到的影像處理觀念實際應用在一個完整的小作品上。從讀取圖片、動漫臉部偵測、自動裁切、銳化降噪，到最後使用 AI 模型進行超解析度放大，希望能讓整個處理流程完整串接，並輸出可以實際使用的動漫角色大頭貼。系統會同時保留「未經 AI 超解析度」與「經過 AI 超解析度」兩個版本，藉此清楚展示傳統 resize 與 AI 上採之間的畫質差異，達到觀察與比較的學習目的。

---

## 三、功能與處理流程

系統使用 OpenCV、Pillow 與 NumPy 進行基本的影像讀取與處理，並透過 dghs-imgutils 函式庫偵測動漫角色的臉部位置，自動裁切出包含頭髮與部分肩膀的頭像範圍（bbox 寬高最大邊外擴 1.6 倍、上方額外往上偏移以容納頭髮）。畫面中若有多張臉，會將所有偵測到的臉部全部處理並逐一輸出；若沒有偵測到任何臉，程式將顯示明確的錯誤訊息並終止執行。

裁切完成後，程式自動進行適度的銳化、降噪與對比度調整，避免動漫線條被過度處理。接著針對每張臉同時產出兩個版本：「raw 版」直接以傳統 resize 縮放至目標尺寸；「sr 版」則根據裁切後尺寸做判斷，若小於目標輸出尺寸便使用 Real-ESRGAN（採用適合動漫圖片的 RealESRGAN_x4plus_anime_6B 權重）進行 AI 超解析度放大，否則同樣直接 resize。兩個版本最終都輸出 1024×1024 像素的 PNG 大頭貼，分別存放於 `outputs/raw/` 與 `outputs/sr/` 兩個子資料夾，且兩邊檔名同步編號，方便對照 AI 上採前後的畫質差異。

**處理流程：**

1. 使用者在終端機執行：`python main.py -i <圖片路徑>`
2. 程式讀取圖片（支援 JPG、JPEG、PNG）
3. dghs-imgutils 偵測動漫臉部，取得畫面上所有臉部的 bbox 座標
4. 對每一張偵測到的臉依序執行下列步驟：
   1. 依 bbox 自動裁切頭像（含頭髮與部分肩膀，外擴 1.6 倍）
   2. 自動進行銳化、降噪、對比度調整
   3. **raw 版**：直接 resize 至 1024×1024
   4. **sr 版**：若裁切後 < 1024px 則跑 Real-ESRGAN 超解析度放大、再 resize 至 1024×1024；若 ≥ 1024px 則同 raw 版直接 resize
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
| PyTorch 2.2.2 | 支援 Real-ESRGAN 模型執行 |

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

安裝 PyTorch（NVIDIA 顯卡版）：

```bash
conda install pytorch=2.2.2 torchvision=0.17.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

安裝其餘套件：

```bash
pip install -r requirements.txt
```

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
C:\Users\<你的帳號>\anaconda3\envs\sbhead\Lib\site-packages\basicsr\data\degradations.py
```

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

看到 `環境驗證通過` 就代表環境設定完成，可以開始執行程式。

---

## 七、執行方式

```bash
conda activate sbhead
python main.py -i sample_images/kasumi.jpg
```

### 輸出位置與命名規則

輸出結果會分別存到 `outputs/raw/` 與 `outputs/sr/` 兩個子資料夾：

- `outputs/raw/output.png`：未做超解析度的版本（僅裁切 + enhance + resize）
- `outputs/sr/output.png`：做完 Real-ESRGAN 超解析度的版本

兩邊檔名一律同步，方便對比 SR 效果。多臉時輸出 N 對檔案（按偵測器原順序）、編號連續；重跑時編號從現有最大值 + 1 開始，不覆蓋既有檔案。

> **注意**：若裁切後尺寸 ≥ 1024，sr 版會跳過 Real-ESRGAN、直接 resize，此時 raw 版與 sr 版內容相同。

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
    ├── raw/                         # 未做 SR 的版本
    └── sr/                          # 做完 SR 的版本（與 raw/ 同名同步）
```
