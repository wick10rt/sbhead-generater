# sbhead-generater — 開發計畫與進度追蹤

> 這份計畫由使用者（張胖胖）制定，Claude 必須**嚴格遵循**，不得擅自調整順序或跳階段。
> 每次 Claude 動工後，必須回報「對應到計畫的哪一步、做了什麼、目前進度」。

---

## 進度標記說明

- `[ ]` 未開始
- `[~]` 進行中
- `[x]` 已完成
- `[!]` 卡住 / 需使用者介入（例如下載權重、放測試圖）

---

## 階段 1：環境準備（先把難裝的搞定）

> 先處理裝環境，因為這是最容易卡住的地方。等程式都寫完才發現裝不起來會很痛。

- [x] 1.1 寫 `requirements.txt`（OpenCV、Pillow、NumPy、PyTorch、realesrgan、basicsr、anime-face-detector）
- [x] 1.2 寫 `.gitignore`（忽略 `weights/*.pth`、`outputs/*`、`intermediate/*`、`__pycache__/`、venv 等）
- [!] 1.3 建 conda 虛擬環境（`conda create -n sbhead python=3.10`）並安裝套件（**使用者本機操作，使用 conda 而非 venv**）
- [!] 1.4 下載 `RealESRGAN_x4plus_anime_6B.pth` 權重放 `weights/`（**使用者本機操作**）
- [!] 1.5 放 1～2 張測試圖到 `sample_images/`（**使用者本機操作**）
- [ ] 1.6 **驗證點**：`python -c "import cv2, torch, realesrgan, anime_face_detector"` 不報錯

---

## 階段 2：建 utils 五個模組（由獨立到串接）

> 每個模組都能單獨 import 測試，不依賴 main.py。

- [ ] 2.1 `utils/avatar_output.py` — `save_avatar(image, output_path, size, border, circle)`
  - 最獨立、最好寫，先做完讓後面有地方輸出
- [ ] 2.2 `utils/enhance.py` — `enhance_image(image)`（銳化＋降噪＋對比）
- [ ] 2.3 `utils/crop_avatar.py` — `crop_by_bbox`、`center_crop`、`manual_crop`
- [ ] 2.4 `utils/face_detect.py` — `detect_largest_face(image) -> bbox or None`
  - 包好 try/except，import 失敗也要能 graceful return None
- [ ] 2.5 `utils/super_resolution.py` — `upscale_image(image, output_size)`
  - 包 Real-ESRGAN 模型載入、CPU/GPU 切換、失敗 fallback 到 `cv2.resize`
- [ ] 2.6 **驗證點**：每個模組寫一個 `if __name__ == "__main__":` 區塊單獨測試

---

## 階段 3：整合 main.py（一次串完所有功能）

- [ ] 3.1 argparse 設好所有參數：`-i / -o / --size / --enhance / --sr / --border / --circle / --crop`
- [ ] 3.2 流程串接：
  - 檢查 input 存在與格式 → 讀圖
  - 臉部偵測（失敗 → fallback 中心裁切；若有 `--crop` 則優先用手動）
  - 依 bbox 擴大裁切 → 正方形
  - 存中間結果到 `intermediate/`
  - 若 `--enhance` → 走 enhance.py
  - 若 `--sr` → 走 super_resolution.py；否則 cv2.resize
  - 依 `--border / --circle` 後製
  - 輸出到 `-o` 指定路徑
- [ ] 3.3 每一步加 print 訊息，讓終端機看得到進度
- [ ] 3.4 錯誤訊息要清楚指出是哪一步死掉

---

## 階段 4：端到端測試（驗證完整版本）

- [ ] 4.1 跑最基本：`python main.py -i sample_images/xxx.jpg -o outputs/test1.png`
- [ ] 4.2 全開：`... --enhance --sr --border --circle`
- [ ] 4.3 故意餵壞圖、不存在的路徑，看錯誤訊息清不清楚
- [ ] 4.4 故意餵沒有臉的圖，確認 fallback 到中心裁切
- [ ] 4.5 故意改名權重檔，確認 SR fallback 到 cv2.resize
- [ ] 4.6 比較不同 `--size` 輸出（512 / 1024 / 2048）

---

## 階段 5：收尾文件與報告

- [ ] 5.1 改寫 `README.md`（安裝、權重下載、用法、所有指令範例、FAQ）
- [ ] 5.2 準備 demo 用的 before/after 對照圖（放進報告）
- [ ] 5.3 寫專題報告（動機、技術、流程、實作、問題、改進、結論）
- [ ] 5.4 `git add . && git commit && git push`

---

## 變更紀錄（Changelog）

> Claude 每次動工後在此追加一筆，格式：`YYYY-MM-DD｜對應步驟｜做了什麼`

- 2026-05-15｜（計畫建立）｜由使用者口述，Claude 寫入此檔案作為進度追蹤基準。
- 2026-05-16｜1.1｜建立 `requirements.txt`，鎖定 numpy<2.0、torchvision<0.18 以避開 basicsr 與 numpy 2.x 已知相容性問題，並在註解中寫明手動修補 `basicsr/data/degradations.py` 的方法。
- 2026-05-16｜1.2｜建立 `.gitignore`，忽略 `weights/*.pth`、`outputs/*`、`intermediate/*`、`sample_images/*`（保留 .gitkeep）、`__pycache__/`、venv、IDE 設定與 OS 雜檔。
- 2026-05-16｜1.3（規範更新）｜使用者指定使用 conda 作為虛擬環境管理工具，已同步更新 `requirements.txt` 安裝指引與本檔 1.3 描述。
