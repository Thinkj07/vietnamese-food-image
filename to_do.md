# LỘ TRÌNH 3 TUẦN HỌC TẬP VÀ THỰC HIỆN DỰ ÁN
# Vietnamese Food Image Classification

> **Dành cho người mới bắt đầu học Machine Learning / Deep Learning**
> **Mục tiêu**: Sau 3 tuần, bạn sẽ nắm vững từ bản chất lý thuyết (toán học trực quan, kiến trúc CNN, Transfer Learning) đến kỹ năng lập trình thực tế (PyTorch, Huấn luyện, Đánh giá, API FastAPI, Web UI, Docker).

---

## TỔNG QUAN LỘ TRÌNH 3 TUẦN

```text
TUẦN 1: Nền tảng Deep Learning cho Xử lý ảnh & Xây dựng Data Pipeline
├── Ngày 1-2: Nền tảng Toán học trực giác & Làm quen với PyTorch Tensors
├── Ngày 3-4: Kiến trúc Mạng nơ-ron tích chập (CNN) từ bản chất
├── Ngày 5-6: Thu thập, Tiền xử lý & Data Augmentation Pipeline
└── Ngày 7: Tổng kết Tuần 1 - Hoàn thiện Module Dataset & Dataloader

TUẦN 2: Huấn luyện Mô hình - Baseline & Transfer Learning / Fine-Tuning
├── Ngày 8-9: Xây dựng Training Loop chuẩn & Huấn luyện Baseline Model
├── Ngày 10-11: Bản chất Transfer Learning & Áp dụng Pretrained Model
├── Ngày 12-13: Kỹ thuật Fine-tuning nâng cao, Scheduler & Regularization
└── Ngày 14: Tổng kết Tuần 2 - Đánh giá mô hình, Confusion Matrix & Error Analysis

TUẦN 3: Xây dựng Ứng dụng Web, REST API & Đóng gói Docker
├── Ngày 15-16: Tối ưu Model Inference & Xây dựng Backend API với FastAPI
├── Ngày 17-18: Thiết kế Giao diện Web UI tương tác (Upload ảnh & Xem kết quả Top-K)
├── Ngày 19-20: Đóng gói toàn bộ ứng dụng bằng Docker & Viết Unit Test
└── Ngày 21: Hoàn thiện tài liệu README.md & Tổng kết nghiệm thu dự án
```

---

# CHI TIẾT LỘ TRÌNH TỪNG NGÀY

---

## 📅 TUẦN 1: NỀN TẢNG DEEP LEARNING & DATA PIPELINE

### 🎯 Mục tiêu tuần 1:
- Hiểu cách biểu diễn hình ảnh số dưới dạng Tensor đa chiều.
- Hiểu bản chất các lớp cơ bản trong mạng CNN (Convolution, Activation, Pooling, Linear).
- Thu thập và tổ chức bộ dữ liệu 5–10 món ăn Việt Nam chuẩn cấu trúc.
- Viết được `Dataset` và `DataLoader` trong PyTorch có áp dụng `Data Augmentation`.

---

### Ngày 1: Nền tảng Toán trực giác & PyTorch Tensors
#### 📚 Kiến thức cần học:
1. **Biểu diễn ảnh trong máy tính**:
   - Ảnh màu RGB = Ma trận 3 chiều: $\text{Height} \times \text{Width} \times 3\text{ Channels}$ với giá trị pixel từ $0 \ đến \ 255$.
   - Tại sao phải chuẩn hóa (Normalize) về dải $[0.0, 1.0]$ hoặc phân phối chuẩn $\mathcal{N}(0, 1)$? (Giúp gradient không bị bùng nổ hoặc biến mất, mô hình hội tụ nhanh hơn).
2. **PyTorch Tensor cơ bản**:
   - Khởi tạo tensor, phép biến đổi hình dạng (`tensor.view()`, `tensor.permute()`, `tensor.squeeze()`, `tensor.unsqueeze()`).
   - Thứ tự chiều trong PyTorch: $[B, C, H, W]$ (Batch size, Channels, Height, Width) khác với OpenCV/Matplotlib là $[H, W, C]$.
3. **Cơ chế tính đạo hàm tự động (Autograd)**:
   - `requires_grad=True`, tính toán Forward $\rightarrow$ hàm mất mát (Loss) $\rightarrow$ `loss.backward()` $\rightarrow$ đạo hàm $\frac{\partial \mathcal{L}}{\partial W}$.

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Thiết lập môi trường Python 3.10+ trên Windows 11 với Git Bash.
- [ ] Cài đặt PyTorch, torchvision, matplotlib, numpy, jupyter.
- [ ] Mở Jupyter Notebook `notebooks/01_data_exploration.ipynb`, viết code tạo Tensor, thực hiện các phép biến đổi Shape và chuyển đổi qua lại giữa ảnh NumPy $\leftrightarrow$ PyTorch Tensor.

#### ❓ Câu hỏi tự kiểm tra (Self-Check):
- *Tại sao PyTorch lại dùng định dạng [B, C, H, W] thay vì [B, H, W, C]?*
- *Phép toán `.unsqueeze(0)` dùng để làm gì khi đưa một bức ảnh đơn lẻ vào model?*

---

### Ngày 2: Luồng học máy cơ bản (Supervised Learning Loop)
#### 📚 Kiến thức cần học:
1. **Hàm mất mát cho phân loại đa lớp (Loss Function)**:
   - Hàm **Softmax**: Biến đổi đầu ra dạng Logits $(z_1, z_2, ..., z_C)$ thành phân phối xác suất $(p_1, p_2, ..., p_C)$ có tổng bằng $1$:
     $$p_i = \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}}$$
   - **Cross-Entropy Loss**: Đo lường sự sai khác giữa phân phối xác suất dự đoán và nhãn thực tế One-Hot:
     $$\mathcal{L}_{CE} = -\sum_{i=1}^C y_i \log(p_i) = -\log(p_{\text{true\_class}})$$
   - *Lưu ý trong PyTorch*: `nn.CrossEntropyLoss()` đã tự động gộp cả Softmax và NLLLoss, do đó đầu ra của model không cần qua tầng `nn.Softmax()`.
2. **Thuật toán tối ưu (Optimizer)**:
   - Gradient Descent, Stochastic Gradient Descent (SGD) với Momentum.
   - Adam / AdamW: Tự động điều chỉnh Learning Rate theo từng trọng số (Adaptive Learning Rate).

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Viết một bài toán phân loại tuyến tính đơn giản (Toy Classification) trên dữ liệu giả lập bằng PyTorch để quen tay với `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.

---

### Ngày 3: Bản chất Mạng nơ-ron tích chập (CNN)
#### 📚 Kiến thức cần học:
1. **Lớp Convolution (Tích chập - `nn.Conv2d`)**:
   - Kernel (Bộ lọc - Filter), Kích thước Kernel ($3 \times 3, 5 \times 5$), Stride (Bước nhảy), Padding (Phần đệm).
   - Công thức tính kích thước đầu ra:
     $$O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$$
   - Ý nghĩa của Feature Map: Các tầng đầu bắt đặc trưng mức thấp (cạnh, góc, texture), các tầng sâu bắt đặc trưng mức cao (hình dạng món ăn, lát thịt, sợi bún).
2. **Hàm kích hoạt phi tuyến tính (Activation Functions)**:
   - `ReLU` ($f(x) = \max(0, x)$), `LeakyReLU`, `GELU`.
3. **Lớp Pooling (`nn.MaxPool2d`, `nn.AvgPool2d`)**:
   - Giảm chiều không gian (Spatial Dimensions), giảm số lượng tham số tính toán và tạo tính bất biến vị trí cục bộ (Translation Invariance).

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Tự viết tay một lớp Convolution nhỏ bằng PyTorch và in ra kích thước Feature Map sau từng bước để hiểu sâu công thức $O$.

---

### Ngày 4: Kiến trúc Mạng CNN Hiện đại (ResNet, MobileNet)
#### 📚 Kiến thức cần học:
1. **Vấn đề suy thoái (Degradation Problem) và Mạng ResNet**:
   - Khi mạng càng sâu, việc lan truyền gradient bị nghẽn $\rightarrow$ Cơ chế **Residual Block (Skip Connection / Shortcut)**: $y = \mathcal{F}(x) + x$.
   - Cho phép mạng học ánh xạ đồng nhất (Identity Mapping), huấn luyện được các mạng rất sâu (ResNet-18, 50, 101).
2. **Kiến trúc tối ưu cho thiết bị nhẹ (MobileNetV3)**:
   - Sử dụng **Depthwise Separable Convolution** giúp giảm tới 80–90% số lượng phép tính so với Convolution thông thường.
   - Thích hợp cho việc triển khai inference nhanh trên CPU của máy tính cá nhân hoặc server nhỏ.

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Import `torchvision.models.resnet50` và `torchvision.models.mobilenet_v3_large`, dùng `torchinfo.summary()` để xem cấu trúc các tầng và số lượng tham số (Parameters).

---

### Ngày 5: Thu thập và Tổ chức Bộ dữ liệu Món ăn Việt Nam
#### 📚 Kiến thức cần học:
1. **Chiến lược dữ liệu trong Machine Learning**:
   - "Garbage In, Garbage Out": Mô hình chỉ tốt khi dữ liệu sạch và đa dạng.
   - Cân bằng số lượng mẫu giữa các lớp (Class Balance) để tránh mô hình bị thiên vị (Bias).
2. **Quy tắc phân chia dữ liệu (Train / Validation / Test Split)**:
   - **Train Set (70%)**: Mô hình trực tiếp học và cập nhật trọng số.
   - **Validation Set (15%)**: Đánh giá sau mỗi epoch để tinh chỉnh Hyperparameter và kích hoạt Early Stopping (mô hình không được học trọng số từ tập này).
   - **Test Set (15%)**: Độc lập hoàn toàn, chỉ đem ra chấm điểm 1 lần cuối cùng sau khi hoàn thành huấn luyện.

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Chọn 5–10 món ăn (ví dụ: `banh_mi`, `bun_bo_hue`, `com_tam`, `pho`, `goi_cuon`).
- [ ] Thu thập hoặc tải bộ dữ liệu ảnh món ăn Việt Nam (tối thiểu 80–100 ảnh / lớp).
- [ ] Viết script Python tự động chia ảnh từ `data/raw/` vào `data/processed/train/`, `val/`, `test/`.
- [ ] Tạo file `data/class_names.json` ánh xạ class index $\leftrightarrow$ tên tiếng Việt.

---

### Ngày 6: Data Augmentation & Xây dựng PyTorch Dataset
#### 📚 Kiến thức cần học:
1. **Data Augmentation (Tăng cường dữ liệu)**:
   - Tại sao cần Augmentation? Giúp chống Overfitting, mô hình học được tính bất biến với góc nghiêng, ánh sáng, vị trí.
   - Các phép biến đổi phổ biến: `RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, `RandomResizedCrop`.
   - *Quy tắc vàng*: Không áp dụng Data Augmentation ngẫu nhiên cho tập `Validation` và `Test`!
2. **`torchvision.transforms.v2` hoặc `transforms` cơ bản**:
   - Normalize theo mean/std của ImageNet.

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Tạo file `src/dataset.py`.
- [ ] Xây dựng hàm `get_transforms(phase='train')`.
- [ ] Viết hàm `create_dataloaders()` sử dụng `torchvision.datasets.ImageFolder` và `torch.utils.data.DataLoader` với các thông số: `batch_size=32`, `shuffle=True`, `num_workers=2`.
- [ ] Viết code vẽ 1 lưới 8 bức ảnh sau khi Augmentation để kiểm tra trực quan.

---

### Ngày 7: Tổng kết Tuần 1 & Hoàn thiện Data Pipeline
#### 🛠️ Việc cần làm (Thực hành):
- [ ] Kiểm tra toàn bộ cấu trúc thư mục dữ liệu `data/processed/`.
- [ ] Kiểm tra `DataLoader` chạy mượt mà, không gặp lỗi định dạng ảnh (như ảnh CMYK, ảnh lỗi Corrupted Image).
- [ ] Viết tài liệu ghi chú các đặc trưng nhận diện chính của từng món vào notebook `01_data_exploration.ipynb`.

---

## 📅 TUẦN 2: HUẤN LUYỆN MÔ HÌNH - BASELINE & TRANSFER LEARNING

### 🎯 Mục tiêu tuần 2:
- Tự viết được Training Loop chuẩn mực trong PyTorch (có Validation, Early Stopping, Model Checkpoint).
- Huấn luyện Baseline Model để lấy mốc so sánh.
- Nắm vững và thực hành Transfer Learning (Feature Extraction & Fine-tuning 2 giai đoạn).
- Đánh giá mô hình toàn diện bằng Accuracy, Precision, Recall, F1-Score và Confusion Matrix.

---

### Ngày 8: Xây dựng Module Mô hình & Training Loop Chuẩn
#### 📚 Kiến thức cần học:
1. **Cấu trúc vòng lặp huấn luyện chuẩn (Standard Training Loop)**:
   - Chuyển chế độ: `model.train()` (bật Dropout, BatchNorm cập nhật running stats) vs. `model.eval()` (tắt Dropout, đóng băng stats BatchNorm).
   - Sử dụng `with torch.no_grad():` trong quá trình Validation/Testing để tiết kiệm bộ nhớ RAM/VRAM và tăng tốc độ.
2. **Theo dõi độ đo (Tracking Metrics)**:
   - Tính Running Loss và Running Accuracy theo từng batch và trung bình theo epoch.

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Tạo file `src/config.py` chứa các siêu tham số: `BATCH_SIZE = 32`, `LR = 1e-3`, `NUM_EPOCHS = 20`, `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`.
- [ ] Tạo file `src/train.py` cài đặt 2 hàm: `train_one_epoch()` và `validate()`.

---

### Ngày 9: Huấn luyện Baseline Model
#### 📚 Kiến thức cần học:
1. **Tại sao cần Baseline?**:
   - Nếu không có Baseline, bạn sẽ không thể chứng minh kỹ thuật Transfer Learning giúp cải thiện bao nhiêu % hiệu năng so với một mô hình cơ bản.
2. **Khái niệm Overfitting & Underfitting**:
   - **Overfitting**: Train Loss rất thấp, Val Loss tăng cao $\rightarrow$ Mô hình "học vẹt".
   - **Underfitting**: Cả Train Loss và Val Loss đều cao $\rightarrow$ Mô hình quá đơn giản hoặc chưa học đủ lâu.

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Trong `src/model.py`, định nghĩa class `SimpleCNN` (3 tầng Conv2D + MaxPool2D + 2 tầng Linear).
- [ ] Huấn luyện `SimpleCNN` trong 15 epochs.
- [ ] Vẽ biểu đồ Train/Val Loss và Train/Val Accuracy qua từng epoch.
- [ ] Lưu lại kết quả Baseline Accuracy để so sánh.

---

### Ngày 10: Bản chất của Transfer Learning (Feature Extraction)
#### 📚 Kiến thức cần học:
1. **Transfer Learning là gì?**:
   - Tận dụng tri thức (Feature Extractors) đã học từ tập dữ liệu khổng lồ (ImageNet với 1.4 triệu ảnh và 1.000 lớp) để áp dụng vào bài toán phân loại món ăn Việt Nam.
2. **Feature Extraction (Phase 1)**:
   - Giữ nguyên toàn bộ trọng số (Weights) của Pretrained Backbone bằng cách đặt `param.requires_grad = False`.
   - Thay thế tầng Fully Connected cuối cùng (`classifier` hoặc `fc`) bằng một Custom Head mới phù hợp với số lớp của bài toán (ví dụ: 10 món ăn).
   - Ưu điểm: Huấn luyện cực nhanh, không lo làm hỏng các đặc trưng tốt đã học sẵn của Backbone.

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Cập nhật `src/model.py` với class `FoodClassifierTransfer`:
  - Khởi tạo backbone: `models.mobilenet_v3_large(weights='DEFAULT')` hoặc `models.resnet50(weights='DEFAULT')`.
  - Đóng băng Backbone.
  - Thay thế tầng phân loại cuối cùng.
- [ ] Huấn luyện 5–8 epochs Phase 1, quan sát tốc độ hội tụ so với Baseline Model.

---

### Ngày 11: Kỹ thuật Fine-Tuning Nâng Cao (Phase 2)
#### 📚 Kiến thức cần học:
1. **Fine-Tuning là gì?**:
   - Sau khi Classification Head đã thích nghi tương đối với bài toán, ta mở băng (`requires_grad = True`) các tầng cuối hoặc toàn bộ Backbone để tinh chỉnh lại các đặc trưng cho khớp hoàn hảo với món ăn Việt Nam.
2. **Tốc độ học phân tầng (Differential Learning Rates)**:
   - Classification Head mới: Dùng Learning Rate bình thường (ví dụ: $1 \times 10^{-4}$).
   - Backbone đã pretrained: Dùng Learning Rate rất nhỏ (ví dụ: $1 \times 10^{-5}$) để tránh hiện tượng "Catastrophic Forgetting" (quên sạch tri thức cũ).
3. **Learning Rate Scheduler**:
   - `CosineAnnealingLR`: Giảm dần Learning Rate theo đồ thị hàm Cosine.
   - `ReduceLROnPlateau`: Tự động giảm Learning Rate khi Val Loss ngừng giảm.

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Cập nhật `src/train.py` hỗ trợ huấn luyện 2 giai đoạn (Phase 1: Feature Extraction $\rightarrow$ Phase 2: Fine-Tuning).
- [ ] Thêm `EarlyStopping` (dừng nếu Val Loss không cải thiện sau 5 epochs) và lưu model tốt nhất vào `models/best_food_model.pt`.

---

### Ngày 12: Đánh giá Mô hình Toàn diện (Evaluation Metrics)
#### 📚 Kiến thức cần học:
1. **Tại sao Accuracy là chưa đủ?**:
   - Khi dữ liệu có sự chênh lệch số lượng mẫu, Accuracy có thể gây ngộ nhận.
2. **Bộ chỉ số phân loại chi tiết**:
   - **Precision**: Trong các mẫu mô hình đoán là "Phở", có bao nhiêu % thực sự là Phở?
   - **Recall**: Trong tất cả các ảnh Phở thực tế, mô hình tìm ra được bao nhiêu %?
   - **F1-Score**: Trung bình điều hòa giữa Precision và Recall.
   - **Macro Average**: Trung bình cộng F1 của tất cả các lớp (đối xử các lớp công bằng nhau).
   - **Top-K Accuracy**: Tỷ lệ nhãn đúng nằm trong Top-3 dự đoán có xác suất cao nhất của mô hình.

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Tạo file `src/evaluate.py`.
- [ ] Load model tốt nhất từ `models/best_food_model.pt`.
- [ ] Chạy inference trên toàn bộ tập `data/processed/test/`.
- [ ] Sử dụng `sklearn.metrics.classification_report` để in bảng chi tiết Precision, Recall, F1 cho từng món ăn.

---

### Ngày 13: Ma trận Nhầm lẫn & Phân tích Lỗi (Error Analysis)
#### 📚 Kiến thức cần học:
1. **Confusion Matrix (Ma trận nhầm lẫn)**:
   - Trục tung: Nhãn thực tế (True Label). Trục hoành: Nhãn dự đoán (Predicted Label).
   - Đường chéo chính thể hiện các mẫu dự đoán đúng. Các ô ngoài đường chéo thể hiện sự nhầm lẫn.
2. **Error Analysis (Phân tích lỗi định tính)**:
   - Xem các bức ảnh bị mô hình dự đoán sai với độ tự tin cao nhất (High Confidence Errors).
   - Tìm hiểu lý do: Do ảnh mờ? Do bát bún bò và phở có màu nước dùng giống nhau? Do góc chụp cận cảnh chỉ thấy lát thịt?

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Dùng `seaborn.heatmap` vẽ và lưu biểu đồ Confusion Matrix vào thư mục `reports/confusion_matrix.png`.
- [ ] Viết script xuất danh sách Top 10 ảnh đoán sai nhất kèm nhãn đúng, nhãn đoán sai và xác suất tương ứng để phân tích.

---

### Ngày 14: Tổng kết Tuần 2 & So sánh Hiệu năng
#### 🛠️ Việc cần làm (Thực hành):
- [ ] Tạo bảng so sánh hiệu năng giữa Baseline Model vs. Transfer Learning Model (Test Accuracy, F1-Score, Training Time, Model Size).
- [ ] Đảm bảo model đạt tiêu chí nghiệm thu (Test Accuracy $\ge 85\%$).
- [ ] Lưu trọng số model cuối cùng vào `models/best_food_model.pt`.

---

## 📅 TUẦN 3: XÂY DỰNG REST API, WEB UI & DOCKER

### 🎯 Mục tiêu tuần 3:
- Đóng gói mô hình thành Class Inference độc lập, tối ưu tốc độ xử lý.
- Xây dựng REST API backend chuẩn với FastAPI.
- Xây dựng giao diện Web UI trực quan, hiện đại bằng HTML/CSS/JS thuần (hỗ trợ kéo thả ảnh, xem Top-3 xác suất).
- Đóng gói ứng dụng chạy bằng Docker và viết tài liệu `README.md` hoàn chỉnh.

---

### Ngày 15: Tối ưu Model Inference & Xây dựng Predictor Module
#### 📚 Kiến thức cần học:
1. **Quy trình Inference trong thực tế**:
   - Nhận ảnh dạng Bytes từ HTTP Request $\rightarrow$ Đọc bằng PIL Image $\rightarrow$ Resize, CenterCrop, Normalize $\rightarrow$ Thêm chiều Batch (`unsqueeze(0)`) $\rightarrow$ `torch.no_grad()` $\rightarrow$ Forward qua model $\rightarrow$ Áp dụng Softmax $\rightarrow$ Lấy Top-K bằng `torch.topk()`.
2. **Tối ưu tốc độ Inference trên CPU**:
   - Set `torch.set_num_threads()` phù hợp.
   - Tránh load lại model mỗi khi có request mới (Load model 1 lần duy nhất khi khởi động server).

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Tạo file `app/predictor.py` chứa class `FoodPredictor` với các phương thức: `__init__(model_path)`, `preprocess(image_bytes)`, `predict(image_bytes, top_k=3)`.
- [ ] Viết script test nhanh `predictor.py` với một ảnh mẫu bất kỳ để đảm bảo thời gian xử lý $< 150\text{ms}$.

---

### Ngày 16: Xây dựng REST API với FastAPI
#### 📚 Kiến thức cần học:
1. **Tại sao chọn FastAPI?**:
   - Hiệu năng cao (dựa trên Starlette và Pydantic), tự động sinh tài liệu tương tác Swagger UI (`/docs`), cú pháp Type-hint rõ ràng.
2. **Cấu trúc Endpoint RESTful**:
   - `GET /health`: Kiểm tra server liveness và model status.
   - `GET /api/v1/classes`: Lấy danh sách các món ăn hỗ trợ.
   - `POST /api/v1/predict`: Nhận file upload (`UploadFile = File(...)`), validate định dạng ảnh và trả về kết quả JSON.

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Tạo file `app/main.py`.
- [ ] Khởi tạo FastAPI app, cấu hình CORS Middleware.
- [ ] Cài đặt các endpoint `/health`, `/api/v1/classes`, `/api/v1/predict`.
- [ ] Khởi chạy server trên Git Bash: `uvicorn app.main:app --reload --port 8000`.
- [ ] Truy cập `http://localhost:8000/docs` để test thử API trực tiếp trên giao diện Swagger.

---

### Ngày 17: Thiết kế Giao diện Web UI (HTML & CSS)
#### 📚 Kiến thức cần học:
1. **Nguyên lý thiết kế giao diện nhận diện hình ảnh**:
   - Trực quan, tối giản, hiển thị vùng kéo thả ảnh (Drag & Drop zone) rõ ràng.
   - Phản hồi trạng thái tức thì (Loading spinner khi đang inference, thông báo lỗi nếu tải file không hợp lệ).
   - Trình bày kết quả trực quan: Ảnh xem trước (Preview), Tên món ăn nổi bật nhất, Thanh đo xác suất (Progress bar) cho Top-3 món ăn có độ tin cậy cao nhất.

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Tạo file `app/templates/index.html`.
- [ ] Tạo file `app/static/css/style.css`.
- [ ] Thiết kế layout responsive, bảng màu hài hòa, kiểu chữ rõ ràng, thanh đo phần trăm xác suất đẹp mắt.

---

### Ngày 18: Tích hợp Frontend với Backend API (JavaScript)
#### 📚 Kiến thức cần học:
1. **Xử lý sự kiện phía Client**:
   - Kéo thả file (`dragover`, `dragleave`, `drop`) và chọn file qua nút bấm (`change`).
   - Sử dụng `FileReader` để hiển thị ảnh xem trước ngay lập tức mà không cần đợi server.
2. **Gọi API bằng `fetch`**:
   - Đóng gói file vào `FormData`.
   - Gửi `POST` request đến `/api/v1/predict`.
   - Nhận JSON phản hồi và cập nhật động DOM (tên món, confidence bar, thời gian xử lý).

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Tạo file `app/static/js/app.js`.
- [ ] Viết logic upload ảnh, gọi API và render danh sách Top-3 kết quả kèm hiệu ứng chuyển động mượt mà.
- [ ] Kiểm thử toàn bộ luồng sử dụng trên trình duyệt (Chrome / Edge).

---

### Ngày 19: Đóng gói Ứng dụng bằng Docker
#### 📚 Kiến thức cần học:
1. **Tại sao cần Docker trong Machine Learning?**:
   - Đảm bảo ứng dụng chạy đồng nhất trên máy tính cá nhân, server hay cloud mà không gặp lỗi lệch phiên bản Python hay thư viện C++.
2. **Tối ưu hóa Dockerfile cho PyTorch**:
   - Sử dụng base image nhẹ: `python:3.10-slim`.
   - Cài đặt PyTorch phiên bản CPU-only trong container để giảm kích thước image từ >5GB xuống còn ~1.2GB.

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Tạo file `requirements.txt` với các thư viện cần thiết đã ghim phiên bản.
- [ ] Tạo file `.dockerignore` (loại trừ `data/raw`, `notebooks/`, `.git`, `__pycache__`).
- [ ] Tạo file `Dockerfile`.
- [ ] Chạy lệnh build và chạy thử container trong Git Bash:
  ```bash
  docker build -t vietnamese-food-classifier:latest .
  docker run -p 8000:8000 vietnamese-food-classifier:latest
  ```
- [ ] Kiểm tra truy cập `http://localhost:8000` từ trình duyệt.

---

### Ngày 20: Viết Unit Tests & Kiểm thử Tự động
#### 📚 Kiến thức cần học:
1. **Testing trong dự án ML**:
   - Test dữ liệu: Kích thước tensor đầu ra của DataLoader có đúng `[B, 3, 224, 224]`?
   - Test mô hình: Output shape của model có đúng `[B, num_classes]`? Giá trị Softmax có tổng bằng 1?
   - Test API: Endpoint `/health` có trả về status 200? Endpoint `/api/v1/predict` có xử lý đúng khi gửi file ảnh hợp lệ và từ chối khi gửi file rỗng?

#### 🛠️ Việc cần làm (Thực hành):
- [ ] Tạo các file test trong thư mục `tests/`:
  - `tests/test_dataset.py`
  - `tests/test_model.py`
  - `tests/test_api.py` (sử dụng `fastapi.testclient.TestClient`).
- [ ] Chạy lệnh `pytest tests/` trên Git Bash và đảm bảo toàn bộ tests đều pass (`100% green`).

---

### Ngày 21: Hoàn thiện Tài liệu README.md & Tổng kết Dự án
#### 🛠️ Việc cần làm (Thực hành):
- [ ] Viết file `README.md` duy nhất thật chuyên nghiệp, bao gồm:
  - Giới thiệu tổng quan dự án và ảnh GIF/Screenshot demo giao diện Web.
  - Bảng tổng kết kết quả độ chính xác (Accuracy, F1-Score so với Baseline).
  - Hướng dẫn cài đặt và chạy môi trường cục bộ (Local Setup).
  - Hướng dẫn chạy bằng Docker.
  - Danh sách API endpoints và curl mẫu.
- [ ] Dọn dẹp code, xóa bỏ các file tạm rác hoặc log thừa.
- [ ] Tự đánh giá lại sản phẩm theo bảng Tiêu chí nghiệm thu (Acceptance Criteria) trong `content.md`.

---

## 💡 MẸO HỌC TẬP VÀ TRÁNH BẪY PHỔ BIẾN CHO NGƯỜI MỚI

1. **Đừng cố học thuộc lòng công thức toán ngay lập tức**: Hãy chú trọng việc hiểu **ý nghĩa trực giác** (ví dụ: Kernel CNN như một chiếc kính lúp quét qua ảnh để tìm đường viền; Learning Rate như độ dài mỗi bước chân khi leo núi trong sương mù).
2. **Luôn kiểm tra kích thước Tensor (`tensor.shape`) sau mỗi bước biến đổi**: 90% lỗi của người mới học PyTorch là do lệch chiều ma trận (`RuntimeError: shape '[...]' is invalid for input of size [...]`). Hãy in `shape` thường xuyên!
3. **Luôn kiểm tra `model.train()` và `model.eval()`**: Quên `model.eval()` khi chạy Validation/Test sẽ khiến kết quả đo lường bị sai lệch do tầng `Dropout` và `BatchNorm` vẫn hoạt động ở chế độ training.
4. **Không bao giờ áp dụng Data Augmentation cho tập Validation và Test**: Việc xoay lật ảnh ngẫu nhiên trong tập Test sẽ làm kết quả đánh giá không còn khách quan và không tái lập được.
5. **Giữ gìn môi trường code sạch sẽ**: Luôn sử dụng Virtual Environment (`venv` hoặc `conda`) và ghi lại chính xác các phiên bản thư viện vào `requirements.txt`.
