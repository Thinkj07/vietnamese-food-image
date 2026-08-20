# ĐẶC TẢ DỰ ÁN CHI TIẾT
# Vietnamese Food Image Classification using Transfer Learning

---

## 1. TỔNG QUAN DỰ ÁN (PROJECT OVERVIEW)

### 1.1. Tên dự án
**Vietnamese Food Image Classification System (Hệ thống phân loại và nhận diện món ăn Việt Nam bằng Deep Learning & Transfer Learning)**

### 1.2. Bối cảnh và Tầm nhìn
Ẩm thực Việt Nam vô cùng phong phú và đặc sắc, tuy nhiên nhiều món ăn có hình thức trình bày và màu sắc tương đối tương đồng (ví dụ: các món bún, phở, mì dùng nước lèo). Dự án này xây dựng một giải pháp Computer Vision toàn diện (End-to-End ML System) nhằm nhận diện tự động món ăn Việt Nam qua hình ảnh chụp, trả về nhãn món ăn cùng độ tin cậy (confidence score) và xếp hạng Top-K dự đoán, phục vụ thông qua Web UI thân thiện và REST API tốc độ cao.

### 1.3. Mục tiêu cốt lõi
1. **Mục tiêu kỹ thuật & sản phẩm**:
   - Xây dựng pipeline xử lý dữ liệu ảnh hoàn chỉnh: từ thu thập, chuẩn hóa, phân chia dataset đến Data Augmentation.
   - Ứng dụng kỹ thuật **Transfer Learning & Fine-tuning** trên các kiến trúc mạng CNN hiện đại (như MobileNetV3, ResNet-50 hoặc EfficientNet-B0) để đạt độ chính xác cao trên tập dữ liệu vừa và nhỏ mà không cần huấn luyện từ đầu (from scratch).
   - Xây dựng REST API bằng **FastAPI** phục vụ inference thời gian thực (latency < 150ms trên CPU).
   - Thiết kế giao diện Web tương tác trực quan (upload/drag-and-drop ảnh, xem biểu đồ xác suất Top-K món ăn).
   - Đóng gói toàn bộ hệ thống bằng **Docker** để dễ dàng triển khai và kiểm thử trên mọi môi trường.

2. **Mục tiêu học tập & Phát triển kỹ năng (Learning Outcomes)**:
   - Hiểu sâu bản chất toán học và trực giác của CNN (Convolution, Pooling, Feature Maps, Receptive Field).
   - Nắm vững cơ chế Transfer Learning (Feature Extraction vs. Fine-tuning, Freeze/Unfreeze backbone layers).
   - Thành thạo PyTorch: `torch.utils.data.Dataset`, `DataLoader`, `torchvision.transforms`, Training Loop, Loss Calculation, Backpropagation, Optimizer (`AdamW`, `SGD`), Learning Rate Scheduler.
   - Thành thạo đánh giá mô hình phân loại đa lớp: Precision, Recall, Macro/Weighted F1-score, Confusion Matrix, Error Analysis.
   - Hiểu cách tối ưu hóa mô hình inference và tích hợp ML vào hệ thống phần mềm thực tế (MLOps cơ bản).

---

## 2. PHẠM VI DỰ ÁN (PROJECT SCOPE)

### 2.1. Trong phạm vi (In-Scope - MVP & Nâng cao)
- **Tập danh mục món ăn (Class Taxonomy)**: Tối thiểu 5–10 lớp món ăn Việt Nam tiêu biểu (ví dụ: *Phở, Bánh mì, Bún chả, Bún bò Huế, Cơm tấm, Gỏi cuốn, Bánh xèo, Chả giò, Cao lầu, Mì Quảng*).
- **Quy trình ML hoàn chỉnh**:
  - Tiền xử lý và Data Augmentation (Random Cropping, Horizontal Flip, Color Jitter, Normalization chuẩn ImageNet).
  - Huấn luyện Baseline Model (Simple CNN hoặc Linear Classifier trên ResNet Features) để làm thước đo so sánh.
  - Huấn luyện Main Model bằng Transfer Learning với Fine-tuning 2 giai đoạn (Frozen Backbone -> Unfrozen Top Layers).
  - Validation trong quá trình huấn luyện kèm Early Stopping và Model Checkpointing (lưu best model theo Val Loss / Val F1).
  - Đánh giá định lượng trên Test Set độc lập.
  - Phân tích lỗi (Error Analysis) và trực quan hóa ma trận nhầm lẫn (Confusion Matrix).
- **Triển khai ứng dụng (Serving & Application)**:
  - Backend REST API cung cấp endpoint `/health`, `/api/v1/predict`, `/api/v1/classes`.
  - Frontend Web UI sạch đẹp, hỗ trợ kéo thả ảnh, hiển thị nhãn dự đoán, thanh đo xác suất (confidence bar chart), thời gian xử lý.
  - Dockerfile và hướng dẫn triển khai môi trường chuẩn (Windows 11 / Linux).

### 2.2. Ngoài phạm vi (Out-of-Scope)
- Nhận diện nhiều đĩa món ăn cùng lúc trong một bức ảnh (Object Detection / YOLO).
- Phân đoạn ngữ nghĩa chi tiết món ăn (Semantic / Instance Segmentation).
- Ước tính kích thước phần ăn, định lượng calo và giá trị dinh dưỡng.
- Nhận diện thành phần nguyên liệu chi tiết trong món ăn.
- Xử lý các món ăn hoàn toàn nằm ngoài tập nhãn huấn luyện (Out-of-Distribution Detection nâng cao).
- Ứng dụng di động native (iOS / Android).

---

## 3. ĐẶC TẢ DỮ LIỆU & DANH MỤC LỚP (DATASET SPECIFICATION)

### 3.1. Danh mục món ăn đề xuất (10 Classes)
| ID | Tên nhãn (Class Name) | Tên tiếng Việt | Đặc điểm nhận diện trực quan chính |
|:---|:---|:---|:---|
| 0 | `banh_mi` | Bánh mì | Ổ bánh mì kẹp thịt, dưa leo, ngò gai, patê, vỏ vàng giòn |
| 1 | `banh_xeo` | Bánh xèo | Vỏ bánh màu vàng nghệ hình bán nguyệt, kèm rau sống, chén nước mắm |
| 2 | `bun_bo_hue` | Bún bò Huế | Nước dùng đỏ cam dầu màu điều, sợi bún to, chả cua, thịt bắp bò, tiết |
| 3 | `bun_cha` | Bún chả | Thịt băm/thịt miếng nướng cháy cạnh ngập trong bát nước chấm, đĩa bún sợi nhỏ |
| 4 | `bun_thit_nuong` | Bún thịt nướng | Tô bún khô rắc đậu phộng, thịt nướng, chả giò, rau thơm, đồ chua |
| 5 | `cha_gio` | Chả giò / Nem rán | Cuốn chiên vàng ruộm, vỏ giòn xốp rế hoặc bánh đa nem |
| 6 | `com_tam` | Cơm tấm | Hạt cơm tấm nhỏ, sườn nướng vân caramen, bì, chả trứng, mỡ hành, đồ chua |
| 7 | `goi_cuon` | Gỏi cuốn | Cuốn bánh tráng trong suốt thấy rõ tôm đỏ, thịt luộc, hẹ xanh |
| 8 | `mi_quang` | Mì Quảng | Sợi mì vàng/trắng to bản, nước nhưỡng ít đậm màu, bánh tráng nướng, trứng cút |
| 9 | `pho` | Phở | Bánh phở trắng dẹt, nước dùng trong vắt, hành hoa xắt nhỏ, thịt bò/gà tái chín |

### 3.2. Cấu trúc lưu trữ dữ liệu
Dataset tuân thủ cấu trúc chuẩn `ImageFolder` của PyTorch:

```text
data/
├── raw/                         # Dữ liệu gốc thu thập
├── processed/
│   ├── train/                   # Tập huấn luyện (70%)
│   │   ├── banh_mi/
│   │   ├── bun_bo_hue/
│   │   └── ... (10 thư mục class)
│   ├── val/                     # Tập kiểm định hyperparameter (15%)
│   │   ├── banh_mi/
│   │   └── ...
│   └── test/                    # Tập đánh giá cuối cùng (15%)
│       ├── banh_mi/
│       └── ...
└── class_names.json             # Danh sách mapping class ID sang nhãn
```

### 3.3. Tiêu chuẩn và chất lượng dữ liệu
- **Số lượng tối thiểu**: 100–200 ảnh / class (tổng cộng 1.000–2.000 ảnh).
- **Định dạng**: JPG, JPEG, PNG.
- **Tiêu chí lọc dữ liệu sạch**:
  - Loại bỏ ảnh mờ, nhòe không rõ nét, ảnh có watermark chèn kín chủ thể.
  - Loại bỏ ảnh trùng lặp (duplicate) hoặc gần trùng lặp.
  - Đảm bảo chủ thể món ăn chiếm ít nhất 50% diện tích khung hình.
  - Đa dạng góc chụp (góc nhìn trên xuống - flat lay, góc nghiêng 45 độ, cận cảnh - close-up), điều kiện ánh sáng khác nhau.

### 3.4. Chiến lược tiền xử lý & Data Augmentation
1. **Pre-processing chuẩn**:
   - Resize ngắn nhất về 256px, sau đó Center Crop về kích thước cố định: $224 \times 224$ (hoặc $256 \times 256$ tùy backbone).
   - Chuyển đổi Pixel sang dải $[0.0, 1.0]$.
   - Normalize theo ImageNet mean & std:
     $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
2. **Augmentation cho Train Set**:
   - `RandomResizedCrop(224, scale=(0.8, 1.0))`
   - `RandomHorizontalFlip(p=0.5)`
   - `RandomRotation(degrees=15)`
   - `ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05)`
3. **Validation & Test Set**:
   - Chỉ áp dụng deterministic transforms: `Resize(256)` $\rightarrow$ `CenterCrop(224)` $\rightarrow$ `ToTensor()` $\rightarrow$ `Normalize()`.

---

## 4. ĐẶC TẢ KỸ THUẬT MÔ HÌNH MACHINE LEARNING (ML SPECIFICATION)

### 4.1. Kiến trúc mô hình
1. **Baseline Model**:
   - Một mạng Custom CNN đơn giản (3 Convolutional blocks + MaxPooling + Dense layers) hoặc ResNet-18 huấn luyện Linear Classifier (freeze toàn bộ backbone).
2. **Main Model (Pretrained Backbones)**:
   - **Primary**: **MobileNetV3-Large** hoặc **ResNet-50** (Pretrained trên ImageNet-1K).
   - **Alternative**: **EfficientNet-B0** (cân bằng hoàn hảo giữa thông số tham số và độ chính xác).
   - **Custom Classification Head**:
     - `AdaptiveAvgPool2d(output_size=1)`
     - `Flatten()`
     - `Linear(in_features=backbone_dim, out_features=256)`
     - `BatchNorm1d(256)`
     - `ReLU()`
     - `Dropout(p=0.3 - 0.4)`
     - `Linear(in_features=256, out_features=num_classes)`

### 4.2. Chiến lược huấn luyện (Transfer Learning Protocol)
Huấn luyện theo chiến lược 2 giai đoạn (2-Phase Training):
- **Phase 1 - Feature Extraction (Warm-up)**:
  - Đóng băng (Freeze) toàn bộ trọng số của Pretrained Backbone (`requires_grad = False`).
  - Chỉ tính toán gradient và cập nhật trọng số cho Custom Classification Head mới khởi tạo.
  - Learning rate: $1 \times 10^{-3}$ với Optimizer `AdamW`. Số epoch: 5–8 epochs.
- **Phase 2 - Fine-Tuning**:
  - Mở băng (Unfreeze) một số block cuối của Backbone (hoặc toàn bộ backbone với Learning Rate cực nhỏ).
  - Learning rate: $1 \times 10^{-4}$ cho Head và $1 \times 10^{-5}$ cho Backbone (Differential Learning Rates).
  - Optimizer: `AdamW` (weight decay = $1 \times 10^{-2}$) hoặc `SGD` (momentum = 0.9).
  - Scheduler: `CosineAnnealingLR` hoặc `ReduceLROnPlateau(patience=3)`.
  - Số epoch: 15–20 epochs kèm Early Stopping nếu Val Loss không cải thiện sau 5 epochs liên tiếp.

### 4.3. Đánh giá và Phân tích lỗi (Evaluation & Error Analysis)
- **Chỉ số đo lường (Metrics)**:
  - **Top-1 Accuracy** & **Top-3 Accuracy**.
  - **Per-class & Macro-averaged Precision, Recall, F1-Score**:
    $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- **Confusion Matrix**: Trực quan hóa ma trận nhầm lẫn bằng `seaborn.heatmap` để xác định rõ cặp nhãn hay bị dự đoán sai nhất (ví dụ: *Phở* nhầm sang *Bún bò Huế* do cùng có nước lèo).
- **Error Analysis Table**: Xuất danh sách các mẫu ảnh dự đoán sai nhất trong Test Set (Top Loss / High Confidence Errors) kèm ảnh gốc để kiểm tra nguyên nhân (nhiễu nền, ánh sáng yếu, góc chụp lạ).

---

## 5. KIẾN TRÚC HỆ THỐNG & ĐẶC TẢ PHẦN MỀM (SYSTEM ARCHITECTURE)

### 5.1. Sơ đồ luồng xử lý (End-to-End Pipeline)

```text
[ Người dùng ] 
      │ (1. Chọn & Upload ảnh món ăn)
      ▼
[ Web UI (HTML5 / Vanilla CSS / JS) ]
      │ (2. Gửi multipart/form-data qua HTTP POST)
      ▼
[ FastAPI Backend (/api/v1/predict) ]
      │ (3. Kiểm tra định dạng, kích thước ảnh)
      │ (4. Chuyển đổi Bytes -> PIL Image -> Tensor)
      │ (5. PyTorch Preprocessing Pipeline: Resize, CenterCrop, Normalize)
      ▼
[ Model Inference Engine (PyTorch / ONNX Runtime) ]
      │ (6. Forward pass -> Logits -> Softmax Function)
      │ (7. Lấy Top-3 Class có xác suất cao nhất)
      ▼
[ FastAPI Response Formatter ]
      │ (8. Trả về JSON: label, confidence, top_k, inference_time_ms)
      ▼
[ Web UI (Render kết quả & Confidence Bar Chart) ]
```

### 5.2. Đặc tả REST API

#### Endpoint 1: Kiểm tra trạng thái hệ thống
- **URL**: `GET /health`
- **Mô tả**: Kiểm tra server backend và trạng thái load mô hình vào bộ nhớ.
- **Response**: `200 OK`
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "model_name": "mobilenet_v3_large_food10",
    "device": "cpu"
  }
  ```

#### Endpoint 2: Danh sách các nhãn món ăn hỗ trợ
- **URL**: `GET /api/v1/classes`
- **Mô tả**: Trả về danh sách tất cả các món ăn mà mô hình có khả năng phân loại.
- **Response**: `200 OK`
  ```json
  {
    "total_classes": 10,
    "classes": [
      {"id": 0, "name": "banh_mi", "display_name": "Bánh mì"},
      {"id": 1, "name": "bun_bo_hue", "display_name": "Bún bò Huế"},
      {"id": 2, "name": "pho", "display_name": "Phở"}
    ]
  }
  ```

#### Endpoint 3: Phân loại hình ảnh món ăn
- **URL**: `POST /api/v1/predict`
- **Content-Type**: `multipart/form-data`
- **Request Body**: `file` (dạng Binary file, định dạng JPG/PNG/WEBP, dung lượng tối đa 10MB).
- **Response**: `200 OK`
  ```json
  {
    "success": true,
    "prediction": {
      "class_id": 9,
      "class_name": "pho",
      "display_name": "Phở",
      "confidence": 0.9542
    },
    "top_k": [
      {
        "class_id": 9,
        "class_name": "pho",
        "display_name": "Phở",
        "confidence": 0.9542
      },
      {
        "class_id": 1,
        "class_name": "bun_bo_hue",
        "display_name": "Bún bò Huế",
        "confidence": 0.0315
      },
      {
        "class_id": 8,
        "class_name": "mi_quang",
        "display_name": "Mì Quảng",
        "confidence": 0.0081
      }
    ],
    "inference_time_ms": 32.4
  }
  ```
- **Error Codes**:
  - `400 Bad Request`: File không đúng định dạng ảnh (không phải JPEG/PNG) hoặc file bị hỏng.
  - `413 Payload Too Large`: Kích thước file vượt quá 10MB.
  - `500 Internal Server Error`: Lỗi xử lý tensor hoặc inference nội bộ.

---

## 6. CẤU TRÚC THƯ MỤC DỰ ÁN (PROJECT DIRECTORY STRUCTURE)

```text
vietnamese-food-image/
├── data/
│   ├── raw/                           # Ảnh thô thu thập ban đầu
│   ├── processed/                     # Dữ liệu đã chia train/val/test
│   └── class_names.json               # Mapping class ID và tên hiển thị
├── notebooks/                         # Jupyter Notebooks nghiên cứu & EDA
│   ├── 01_data_exploration.ipynb      # Khám phá và phân tích dataset
│   ├── 02_baseline_model.ipynb        # Huấn luyện baseline model
│   ├── 03_transfer_learning.ipynb     # Transfer learning & fine-tuning
│   └── 04_error_analysis.ipynb        # Đánh giá chi tiết & Confusion Matrix
├── src/                               # Source code chính của dự án
│   ├── __init__.py
│   ├── config.py                      # Thiết lập Hyperparameters & Paths
│   ├── dataset.py                     # PyTorch Dataset, Data Transforms & Loaders
│   ├── model.py                       # Định nghĩa kiến trúc mạng Transfer Learning
│   ├── train.py                       # Training loop, validation & model saving
│   ├── evaluate.py                    # Script tính metrics & xuất confusion matrix
│   └── utils.py                       # Hàm bổ trợ (logging, plotting, metrics)
├── app/                               # Ứng dụng Web & API Serving
│   ├── __init__.py
│   ├── main.py                        # FastAPI entry point
│   ├── predictor.py                   # Class load model & chạy inference
│   ├── static/                        # CSS, JS, Assets cho Web UI
│   │   ├── css/style.css
│   │   └── js/app.js
│   └── templates/                     # HTML Views
│       └── index.html
├── models/                            # Nơi lưu trữ checkpoint mô hình
│   └── best_food_model.pt             # PyTorch model weights tối ưu nhất
├── tests/                             # Unit tests & Integration tests
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── content.md                         # Đặc tả dự án chi tiết
├── Dockerfile                         # Dockerfile đóng gói ứng dụng
├── README.md                          # Tài liệu hướng dẫn sử dụng dự án duy nhất
├── requirements.txt                   # Danh sách thư viện Python
└── to_do.md                           # Lộ trình học tập và thực hành 3 tuần
```

---

## 7. TIÊU CHÍ NGHIỆM THU DỰ ÁN (ACCEPTANCE CRITERIA & KPIS)

| Hạng mục | Tiêu chí đánh giá (KPI) | Mức độ tối thiểu đạt chuẩn | Mức độ xuất sắc (Target) |
|:---|:---|:---|:---|
| **Dữ liệu** | Số lượng lớp món ăn | $\ge 5$ món | $8 - 10$ món |
| | Số lượng ảnh mỗi lớp | $\ge 80$ ảnh/lớp | $150 - 200$ ảnh/lớp |
| | Tỷ lệ phân chia tập dữ liệu | 70% Train - 15% Val - 15% Test | Phân tầng (Stratified split) |
| **Model ML** | Test Set Accuracy | $\ge 80.0\%$ | $\ge 88.0\%$ |
| | Macro F1-Score | $\ge 0.78$ | $\ge 0.85$ |
| | So sánh Baseline | Có bảng so sánh rõ ràng | Vượt trội baseline ít nhất $10\%$ F1 |
| **Hệ thống** | Thời gian Inference (CPU) | $< 250\text{ ms / request}$ | $< 100\text{ ms / request}$ |
| | Web UI | Upload ảnh, hiển thị Top-1 | Kéo thả, Top-3 Bar Chart, Loading spinner |
| | Dockerization | Chạy `docker build` & `run` thành công | Tối ưu kích thước image (< 1.5GB) |
| **Chất lượng code**| Cấu trúc code | Tách module rõ ràng, có file config | Viết Type hint đầy đủ, có unit tests cơ bản |