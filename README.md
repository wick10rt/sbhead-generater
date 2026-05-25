# sbhead-generater
**Bang Dream 角色大頭貼自動生成系統**

輸入一張動漫角色圖片，自動偵測畫面中**所有**角色臉部、裁切頭像、畫質增強、AI 超解析度放大，每張臉同時輸出「未做 SR」與「做完 SR」兩個版本的 1024×1024 PNG 大頭貼，方便對比畫質差異。

---

## 你需要手動完成的步驟

> 程式碼由 Claude 負責，以下五件事需要你在本機操作完成。

---

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

---

### 步驟 2：下載 Real-ESRGAN 權重檔

前往以下頁面下載 `RealESRGAN_x4plus_anime_6B.pth`：

> https://github.com/xinntao/Real-ESRGAN/releases

下載後放到專案的 `weights/` 資料夾：

```
sbhead-generater/
└── weights/
    └── RealESRGAN_x4plus_anime_6B.pth   ← 放這裡
```

---

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

---

### 步驟 4：放入測試圖片

將 1～2 張 Bang Dream 角色圖片放到 `sample_images/` 資料夾（JPG 或 PNG 均可）。

```
sbhead-generater/
└── sample_images/
    └── kasumi.jpg   ← 放這裡
```

> 注意：`sample_images/` 已加入 `.gitignore`，圖片不會被上傳到 GitHub。

---

### 步驟 5：驗證環境

所有步驟完成後，執行以下指令確認沒有報錯：

```bash
python -c "import cv2, torch, realesrgan; from imgutils.detect import detect_faces; print('環境驗證通過')"
```

看到 `環境驗證通過` 就代表環境設定完成，可以開始執行程式。

---

## 執行方式

```bash
conda activate sbhead
python main.py -i sample_images/kasumi.jpg
```

### 輸出位置與命名規則

輸出結果會分別存到 `outputs/raw/` 與 `outputs/sr/` 兩個子資料夾：

- `outputs/raw/output.png`：未做超解析度的版本（僅裁切 + enhance + resize）
- `outputs/sr/output.png`：做完 Real-ESRGAN 超解析度的版本

兩邊檔名一律同步，方便對比 SR 效果。

**多臉時**：偵測到 N 張臉就輸出 N 對檔案（按偵測器原順序、單張失敗會跳過繼續處理其他臉），編號連續：

```
outputs/
├── raw/
│   ├── output.png       ← 第一張臉
│   ├── output(1).png    ← 第二張臉
│   └── output(2).png    ← 第三張臉
└── sr/
    ├── output.png       ← 與 raw/output.png 同一張臉
    ├── output(1).png
    └── output(2).png
```

重跑時編號從 raw/ 與 sr/ 兩邊現有最大編號 + 1 開始，不會覆蓋既有檔案。

> **注意**：若裁切後尺寸 ≥ 1024，sr 版會跳過 Real-ESRGAN、直接 resize，此時 raw 版與 sr 版內容相同。

---

## 專案結構

```
sbhead-generater/
├── main.py                          # 主程式
├── requirements.txt                 # 套件清單
├── PLAN.md                          # 開發進度追蹤
├── 計畫書.md                        # 給教授看的計畫書
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
