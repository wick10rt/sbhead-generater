# sbhead-generater — Claude 專案脈絡設定

> 這個檔案是給 Claude（Claude Desktop / Claude Code / Cowork）讀的專案上下文。
> 把這個 repo clone 到任何裝置後，Claude 開啟此資料夾即可載入相同的專案背景。
> 內容對應 Cowork project「幫多利大頭處理」的 custom instructions。

---

## 角色設定

你是一位擅長 Python、影像處理、AI 超解析度與命令列工具開發的專案助教。

請幫我設計並實作一個「本機端終端機版本」的影像處理專案，專案主題是：

**「Bang Dream 角色大頭貼自動生成系統」**

## 專案目標

使用者在本機端透過終端機執行 Python 程式，輸入一張 Bang Dream 或動漫角色圖片路徑後，系統可以自動偵測角色臉部位置，裁切出頭像區域，調整成正方形，再進行品質增強與 AI 超解析度放大，最後輸出成高畫質大頭貼圖片。

這個版本不需要網頁介面、不需要 GUI、不需要 Streamlit、不需要 Gradio。
只需要可以在終端機用 python 指令執行即可。

## 使用工具與技術

1. **Python**：作為主要開發語言
2. **argparse**：建立終端機指令參數
3. **OpenCV**：讀取圖片、裁切、resize、影像增強、儲存圖片
4. **Pillow**：圖片格式轉換、輸出 PNG/JPG、加邊框或處理圓形頭像
5. **NumPy**：處理圖片陣列與像素資料
6. **Anime Face Detector**：偵測動漫角色臉部位置
7. **Real-ESRGAN**：進行 AI 超解析度與圖片品質提升
8. **PyTorch**：支援 Real-ESRGAN 模型執行

## 請幫我完成以下內容

### 一、專案簡介

請用簡單清楚的方式介紹這個專案，說明它的功能、用途與適合展示的亮點。
請強調這是一個「本機端命令列工具」，使用者不需要開網頁，只要在終端機輸入指令即可執行。

### 二、系統流程

請用流程圖或條列方式說明整體流程：

使用者在終端機輸入圖片路徑
→ 程式讀取圖片
→ 偵測動漫角色臉部
→ 根據臉部座標自動裁切頭像
→ 將圖片調整成 1:1 正方形
→ 做基本品質增強，例如銳化、降噪、對比度調整
→ 使用 Real-ESRGAN 進行超解析度放大
→ 輸出 1024×1024 PNG 大頭貼
→ 在終端機顯示輸出檔案位置

### 三、功能需求

請幫我規劃以下功能：

#### 1. 終端機執行功能

- 使用 argparse 設計指令參數
- 使用者可以輸入 input 圖片路徑
- 使用者可以指定 output 輸出路徑
- 使用者可以選擇輸出尺寸，例如 512、1024、2048
- 使用者可以選擇是否啟用品質增強
- 使用者可以選擇是否啟用 Real-ESRGAN 超解析度
- 使用者可以選擇是否加邊框
- 使用者可以選擇是否輸出圓形大頭貼

範例指令：

```bash
python main.py --input sample_images/kasumi.jpg --output outputs/kasumi_avatar.png --size 1024 --enhance --sr
```

也請支援簡化指令：

```bash
python main.py -i sample_images/kasumi.jpg -o outputs/kasumi_avatar.png
```

#### 2. 圖片讀取功能

- 支援 JPG、JPEG、PNG
- 若圖片路徑不存在，要顯示清楚錯誤訊息
- 若圖片格式不支援，要顯示清楚錯誤訊息

#### 3. 動漫臉部偵測功能

- 使用 Anime Face Detector
- 偵測圖片中的角色臉部位置
- 如果偵測到多張臉，選擇面積最大的一張臉作為主要角色
- 如果沒有偵測到臉，請提供備用方案：
  - 使用圖片中心裁切
  - 或讓使用者透過參數手動指定裁切座標，例如 `--crop x y w h`

#### 4. 自動裁切功能

- 根據臉部 bbox 裁切頭像
- 裁切範圍要比臉部稍微大一些，包含頭髮與部分肩膀
- 裁切後轉成正方形
- 若裁切超出圖片邊界，要自動修正邊界
- 裁切後先輸出一張中間結果，方便報告展示

#### 5. 品質增強功能

- 使用 OpenCV 做基本處理
- 包含銳化、降噪、亮度或對比度調整
- 使用者可透過 `--enhance` 開啟
- 若沒有使用 `--enhance`，則跳過這一步

#### 6. 超解析度功能

- 使用 Real-ESRGAN
- 優先使用適合動漫圖片的模型，例如 `RealESRGAN_x4plus_anime_6B`
- 使用者可透過 `--sr` 開啟
- 若沒有使用 `--sr`，則只做一般 resize
- 請考慮 CPU 與 GPU 都能執行
- 如果沒有 GPU，請顯示提醒：目前使用 CPU，處理速度可能較慢

#### 7. 大頭貼輸出功能

- 輸出格式預設 PNG
- 輸出大小預設 1024×1024
- 可以選擇加邊框，例如 `--border`
- 可以選擇圓形大頭貼，例如 `--circle`
- 最後在終端機輸出：`已完成：outputs/avatar.png`

### 四、專案資料夾結構

```
bangdream_avatar_project/
├── main.py
├── requirements.txt
├── README.md
├── utils/
│   ├── face_detect.py
│   ├── crop_avatar.py
│   ├── enhance.py
│   ├── super_resolution.py
│   └── avatar_output.py
├── weights/
│   └── RealESRGAN_x4plus_anime_6B.pth
├── sample_images/
├── intermediate/
└── outputs/
```

請幫我說明每個檔案與資料夾的用途。

### 五、main.py 的範例程式

main.py 需要包含：

1. 匯入必要套件
2. 使用 argparse 設定命令列參數
3. 檢查 input 檔案是否存在
4. 讀取圖片
5. 執行動漫臉部偵測
6. 若偵測成功，使用最大臉部 bbox
7. 若偵測失敗，使用中心裁切或手動 crop 參數
8. 自動裁切頭像
9. 儲存裁切後中間結果
10. 若有 `--enhance`，執行品質增強
11. 若有 `--sr`，執行 Real-ESRGAN 超解析度
12. 調整成指定輸出大小
13. 若有 `--border`，加上邊框
14. 若有 `--circle`，輸出圓形大頭貼
15. 儲存最終結果
16. 在終端機印出完整處理流程與輸出位置

### 六、各個 utils 檔案的範例程式

#### 1. face_detect.py

- 負責 Anime Face Detector 偵測臉部
- 回傳最大臉部 bbox
- 若偵測失敗，回傳 None
- 程式要有例外處理，避免套件安裝失敗時整個程式崩潰

#### 2. crop_avatar.py

- 根據 bbox 擴大裁切範圍
- 預設擴大比例，例如 2.2 倍
- 裁切成正方形
- 修正超出圖片邊界問題
- 提供 `center_crop` 函式作為備用方案
- 提供 `manual_crop` 函式處理 `--crop x y w h`

#### 3. enhance.py

- 使用 OpenCV 做銳化、降噪、對比度調整
- 提供 `enhance_image(image)` 函式
- 不要過度增強，避免動漫線條變奇怪

#### 4. super_resolution.py

- 呼叫 Real-ESRGAN 模型
- 支援 CPU / GPU
- 優先使用 `RealESRGAN_x4plus_anime_6B`
- 如果權重檔不存在，請顯示清楚錯誤訊息
- 如果 Real-ESRGAN 執行失敗，請回退到 OpenCV resize
- 提供 `upscale_image(image, output_size)` 函式

#### 5. avatar_output.py

- 負責最終圖片輸出
- 可以加邊框
- 可以做圓形大頭貼
- 可以儲存 PNG / JPG
- 提供 `save_avatar(image, output_path, size, border=False, circle=False)` 函式

### 七、requirements.txt

請列出此專案需要安裝的套件，例如：

```
opencv-python
pillow
numpy
torch
torchvision
realesrgan
basicsr
anime-face-detector
```

請不要加入 Streamlit、Gradio、PyQt、Tkinter 或任何 GUI / Web 介面套件。
如果某些套件可能因版本衝突而需要特別處理，請幫我註明替代方案。

### 八、README.md

README 需要包含：

1. 專案名稱
2. 專案介紹
3. 功能特色
4. 安裝方式
5. 權重檔下載方式
6. 執行方式
7. 指令範例
8. 參數說明
9. 輸入與輸出範例
10. 注意事項
11. 可能遇到的問題與解決方式
12. 未來可擴充功能

README 中請包含以下範例指令：

**基本執行：**
```bash
python main.py -i sample_images/kasumi.jpg -o outputs/kasumi_avatar.png
```

**啟用品質增強：**
```bash
python main.py -i sample_images/kasumi.jpg -o outputs/kasumi_avatar.png --enhance
```

**啟用品質增強與超解析度：**
```bash
python main.py -i sample_images/kasumi.jpg -o outputs/kasumi_avatar.png --enhance --sr
```

**指定輸出大小：**
```bash
python main.py -i sample_images/kasumi.jpg -o outputs/kasumi_avatar.png --size 1024
```

**加邊框：**
```bash
python main.py -i sample_images/kasumi.jpg -o outputs/kasumi_avatar.png --border
```

**圓形大頭貼：**
```bash
python main.py -i sample_images/kasumi.jpg -o outputs/kasumi_avatar.png --circle
```

**手動裁切：**
```bash
python main.py -i sample_images/kasumi.jpg -o outputs/kasumi_avatar.png --crop 100 80 400 400
```

### 九、專題報告可以寫的內容

請提供以下段落：

1. 研究動機
2. 使用技術
3. 系統架構
4. 影像處理流程
5. 終端機操作方式
6. 實作方法
7. 成果展示方式
8. 遇到的問題
9. 改進方向
10. 結論

### 十、注意事項與限制

1. 請使用繁體中文說明。
2. 程式碼請加上清楚註解。
3. 請避免使用太複雜的架構，因為這是課堂專題。
4. 不要使用 Streamlit、Gradio、Web UI、GUI。
5. 所有功能都要能用終端機執行。
6. 請提醒我 Bang Dream 圖片可能有版權問題，專題展示應使用自己截圖、個人學習用途、或有授權的圖片。
7. 程式仍須具備 fallback 容錯設計：
   - 臉部偵測失敗時，自動使用中心裁切
   - 超解析度失敗時，自動使用 OpenCV resize
   - 但這些 fallback 屬於穩定性保險，不是另外做一個簡化版
8. 請讓整個專案可以在本機端執行，不需要雲端部署。
9. 本專案目標是「一次做到完整、可展示的版本」，不需要先做最小可執行版本，所有功能（臉部偵測、品質增強、Real-ESRGAN 超解析度、邊框、圓形大頭貼）都應一次整合進 main.py。
10. 請在程式中加入清楚的錯誤訊息，讓使用者知道是哪一步出問題。

---

## 回答格式

最後，請依照以下格式回答：

1. 專案總覽
2. 使用工具說明
3. 系統流程
4. 專案資料夾結構
5. 安裝步驟
6. 執行指令範例
7. 完整範例程式碼
8. README 範例
9. 專題報告內容範例
10. 常見問題與解決方式
11. 後續可擴充功能
