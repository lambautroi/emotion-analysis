# BÁO CÁO BÀI TẬP LỚN

## CHUYÊN ĐỀ HỆ THỐNG THÔNG TIN

---

# ĐỀ TÀI: HỆ THỐNG PHÂN TÍCH CẢM XÚC BÌNH LUẬN SỬ DỤNG MACHINE LEARNING VÀ GEMINI AI

---

GVHD: [Tên giảng viên]

Sinh viên thực hiện: [Họ tên] - [MSSV]

Lớp: [Tên lớp]

Học kỳ: Học kỳ [X] - Năm học 20XX-20XX

---

# MỤC LỤC

1. [Giới thiệu chung](#1-giới-thiệu-chung)
2. [Cơ sở lý thuyết](#2-cơ-sở-lý-thuyết)
3. [Phương pháp đề xuất](#3-phương-pháp-đề-xuất)
4. [Kết quả thực nghiệm](#4-kết-quả-thực-nghiệm)
5. [Kết luận](#5-kết-luận)

---

# 1. Giới thiệu chung

## 1.1. Lý do chọn đề tài

Trong thời đại kỷ nguyên số, mạng xã hội và các nền tảng thương mại điện tử đã trở thành phần không thể thiếu trong cuộc sống hàng ngày. Hàng triệu bình luận, đánh giá và phản hồi được tạo ra mỗi ngày trên các nền tảng như Facebook, Shopee, Lazada, TikTok,... Những bình luận này chứa đựng một lượng lớn thông tin có giá trị về cảm xúc, ý kiến và mong muốn của khách hàng đối với sản phẩm hoặc dịch vụ.

Việc phân tích thủ công hàng nghìn, hàng triệu bình luận là bất khả thi về mặt thời gian và nhân lực. Do đó, nhu cầu về một hệ thống tự động có khả năng phân tích cảm xúc (Sentiment Analysis) từ văn bản trở nên cấp thiết hơn bao giờ hết. Mặc dù các API AI hiện đại như Gemini mang lại độ chính xác cao, nhưng chúng gặp phải những rào cản về kết nối Internet, độ trễ và chi phí/giới hạn quota (Rate limit). Vì vậy, một giải pháp kết hợp giữa các mô hình Học máy truyền thống (Machine Learning) xử lý offline tốc độ cao và AI hiện đại là phương pháp tiếp cận tối ưu.

## 1.2. Mục tiêu nghiên cứu

Đề tài này hướng đến việc xây dựng một hệ thống phân tích cảm xúc bình luận tự động với các mục tiêu cụ thể:

1. **Huấn luyện mô hình Học máy (Machine Learning):** Xây dựng và đánh giá các mô hình ML truyền thống (Naive Bayes, Logistic Regression, SVM, Random Forest) trên bộ dữ liệu tiếng Việt chuẩn.
2. **So sánh hiệu năng:** Đối chiếu hiệu quả, độ chính xác và tốc độ giữa mô hình ML tự huấn luyện (đặc biệt là SVM) với Google Gemini AI API.
3. **Phân tích cảm xúc tự động:** Nhận diện và phân loại cảm xúc trong bình luận thành ba loại: tích cực (positive), trung tính (neutral), và tiêu cực (negative).
4. **Xử lý hàng loạt Offline & Online:** Hỗ trợ phân tích nhiều bình luận cùng lúc thông qua việc upload file CSV/Excel với tốc độ cao bằng mô hình ML offline, đồng thời vẫn giữ tùy chọn phân tích online qua Gemini.
5. **Giao diện trực quan:** Cung cấp giao diện web thân thiện, dễ sử dụng cho người dùng.

## 1.3. Phạm vi nghiên cứu

- Ngôn ngữ: Tiếng Việt.
- Lĩnh vực: Bình luận đánh giá sản phẩm, dịch vụ.
- Bộ dữ liệu huấn luyện: UIT-VSFC (Vietnamese Students' Feedback Corpus).
- Công nghệ: Machine Learning (Scikit-Learn), NLP (Underthesea), Google Gemini AI API, Flask Framework.

## 1.4. Kết quả đạt được

- Huấn luyện thành công mô hình Support Vector Machine (SVM) đạt độ chính xác ~89.39%.
- Xây dựng thành công ứng dụng web tích hợp song song cả Machine Learning (Offline) và Gemini AI (Online).
- Hỗ trợ phân tích đơn lẻ và hàng loạt với tốc độ nhanh chóng.
- Hệ thống thống kê và quản lý dữ liệu hiệu quả qua SQLite.

---

# 2. Cơ sở lý thuyết

## 2.1. Phân tích cảm xúc (Sentiment Analysis)

Phân tích cảm xúc (Sentiment Analysis) hay còn gọi là Opinion Mining là một lĩnh vực con của Xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP), tập trung vào việc nhận diện, trích xuất và phân loại cảm xúc được biểu đạt trong văn bản (Positive, Neutral, Negative).

## 2.2. Xử lý Ngôn ngữ Tự nhiên (NLP) tiếng Việt

Khác với tiếng Anh (từ được phân tách bằng khoảng trắng), tiếng Việt có đặc thù là các từ ghép tạo thành từ nhiều âm tiết (ví dụ: "chất lượng", "sinh viên"). Để máy tính hiểu đúng ngữ nghĩa, bước **Tách từ (Word Segmentation)** là bắt buộc. Trong đề tài này, thư viện `underthesea` được sử dụng để gom các cụm âm tiết có nghĩa lại với nhau, giúp mô hình ML không hiểu sai văn cảnh.

## 2.3. Rút trích đặc trưng văn bản (TF-IDF)

**TF-IDF** (Term Frequency - Inverse Document Frequency) là một kỹ thuật chuyển đổi văn bản thành các vector số học:
- **TF (Tần suất xuất hiện):** Đo lường số lần một từ xuất hiện trong một bình luận.
- **IDF (Nghịch đảo tần suất văn bản):** Đánh giá mức độ quan trọng của từ đó. Các từ xuất hiện ở mọi câu (ví dụ: "là", "thì", "mà") sẽ bị giảm trọng số. Ngược lại, các từ mang ý nghĩa cảm xúc mạnh ("tuyệt_vời", "tệ_hại") sẽ có trọng số cao.

## 2.4. Các thuật toán Machine Learning (Học máy)

Hệ thống tiến hành thực nghiệm và so sánh 4 thuật toán phân loại kinh điển:
1. **Naive Bayes:** Thuật toán dựa trên định lý Bayes, tính toán xác suất độc lập của các từ để đưa ra dự đoán. Rất nhanh và hiệu quả với văn bản.
2. **Logistic Regression:** Mô hình hồi quy phân loại tuyến tính, tìm ra ranh giới quyết định (decision boundary) bằng cách áp dụng hàm sigmoid/softmax để tính xác suất cho các lớp cảm xúc.
3. **Support Vector Machine (SVM):** Thuật toán tìm ra một "siêu mặt phẳng" (hyperplane) có khoảng cách (margin) lớn nhất để phân tách giữa các nhãn cảm xúc. Rất mạnh mẽ khi xử lý không gian vector TF-IDF nhiều chiều.
4. **Random Forest:** Thuật toán học tập kết hợp (ensemble learning), tạo ra nhiều cây quyết định (decision trees) ngẫu nhiên và tổng hợp kết quả (voting) để tăng độ chính xác, tránh overfitting.

## 2.5. Google Gemini AI

Google Gemini là mô hình AI đa phương thức được phát triển bởi Google DeepMind. Trong dự án, Gemini 2.5 Flash được sử dụng như một mốc chuẩn mực (Baseline) đại diện cho sức mạnh của LLM (Large Language Models) không cần huấn luyện (Zero-shot) nhờ khả năng hiểu sâu ngữ cảnh tự nhiên.

## 2.6. Flask Web Framework & SQLite

- **Flask:** Một micro web framework bằng Python, rất nhẹ và linh hoạt, dùng để xây dựng RESTful API và render giao diện HTML/CSS.
- **SQLite:** Hệ quản trị cơ sở dữ liệu quan hệ nhẹ, được tích hợp sẵn trong Python, dùng để lưu trữ lịch sử và kết quả phân tích bình luận.

---

# 3. Phương pháp đề xuất

## 3.1. Quy trình huấn luyện mô hình Machine Learning

Hệ thống được phát triển theo quy trình chuẩn của một bài toán NLP:

1. **Thu thập dữ liệu:** Sử dụng bộ dữ liệu **UIT-VSFC** (~16,000 bình luận tiếng Việt). Dữ liệu được chia thành tập Train (huấn luyện) và tập Test (đánh giá).
2. **Tiền xử lý văn bản:** 
   - Chuyển thành chữ in thường (lowercase).
   - Xóa bỏ dấu câu, ký tự đặc biệt.
   - Tách từ tiếng Việt bằng `underthesea`.
3. **Vector hóa:** Chuyển đổi văn bản đã tiền xử lý sang không gian vector bằng **TF-IDF Vectorizer** với max_features=10000.
4. **Huấn luyện & Chọn lọc:** Đưa ma trận TF-IDF vào huấn luyện cùng lúc 4 thuật toán (NB, LR, SVM, RF). Lựa chọn mô hình đạt Accuracy và F1-score cao nhất để trích xuất file `best_model.pkl`.

## 3.2. Kiến trúc hệ thống tổng quan

```text
┌─────────────┐     ┌──────────────┐      ┌─────────────────────────┐
│   User      │────▶│   Flask      │──┬──▶│   Machine Learning      │
│   Browser   │◀────│   Server     │  │   │   (SVM Model - Offline) │
└─────────────┘     └──────────────┘  │   └─────────────────────────┘
                           │          │   ┌─────────────────────────┐
                           ▼          └──▶│   Google Gemini API     │
                    ┌──────────────┐      │   (Online LLM)          │
                    │   SQLite     │      └─────────────────────────┘
                    │   Database   │
                    └──────────────┘
```

## 3.3. Thiết kế giao diện người dùng (UI)

Trang chủ cung cấp trải nghiệm thân thiện với:
- **Thống kê:** Các thẻ hiển thị tổng số bình luận, tỷ lệ phần trăm Tích cực/Trung tính/Tiêu cực.
- **Dropdown chọn Model:** Cho phép người dùng chuyển đổi linh hoạt giữa "Google Gemini API" và "Machine Learning - SVM".
- **Nhập thủ công:** Box text cho phép dán bình luận để phân tích nhanh.
- **Tải lên hàng loạt (Batch):** Box upload file CSV/Excel. Rất phù hợp kết hợp cùng Model ML để xử lý hàng nghìn dòng chỉ trong vài giây.
- **Bảng kết quả:** Lưu trữ lịch sử phân tích với nhãn màu dễ nhìn.

## 3.4. Luồng xử lý nghiệp vụ

**Phân tích bình luận (Đơn lẻ & Hàng loạt):**
1. User chọn phương thức nhập liệu (Text hoặc File) và chọn loại Model (ML hoặc Gemini).
2. Flask Server tiếp nhận. Nếu là file, hệ thống sẽ đưa vào Background Thread (xử lý nền) để tránh đơ giao diện.
3. Tùy thuộc vào Model:
   - **Machine Learning:** Chuỗi văn bản đi qua bộ `preprocess_text`, chuyển thành vector qua `tfidf_vectorizer`, sau đó dự đoán qua `svm_model.predict()`. Quá trình diễn ra offline, phản hồi tính bằng mili-giây.
   - **Gemini API:** Hệ thống gói bình luận vào Prompt chuẩn hóa và gửi Request lên Google Servers, chờ lấy phản hồi JSON.
4. Kết quả được bóc tách (Nhãn cảm xúc, Điểm số độ tin cậy) và lưu vào `SQLite`.
5. Người dùng xem trực tiếp trên bảng điều khiển Web.

## 3.5. Cấu trúc mã nguồn huấn luyện Modular

Để thuận tiện cho việc đánh giá, theo dõi và bảo trì, toàn bộ quá trình huấn luyện Machine Learning được nhóm thiết kế dưới dạng **Modular Scripts** độc lập:
- `1_prepare_data.py`: Đảm nhiệm việc tải dữ liệu, tiền xử lý và chuyển hóa thành Vector, tạo ra không gian dữ liệu chuẩn duy nhất (`data_prepared.pkl`).
- Các file `2_train_nb.py`, `3_train_lr.py`, `4_train_rf.py`, `5_train_svm.py`: Tách biệt hoàn toàn việc huấn luyện từng thuật toán riêng lẻ. Tính module hóa này giúp dễ dàng thay đổi tham số hoặc thuật toán mà không ảnh hưởng toàn hệ thống.
- `6_evaluate_models.py`: Đóng vai trò tổng hợp kết quả của 4 mô hình trên, kết hợp với sức mạnh của Gemini để xuất ra các bảng so sánh và biểu đồ trực quan một cách nhanh chóng.

---

# 4. Kết quả thực nghiệm

## 4.1. Môi trường thử nghiệm

- **Dataset:** UIT-VSFC (11,426 mẫu Train, 3,166 mẫu Test).
- **Phần cứng:** PC tiêu chuẩn (Intel Core i5, 8GB RAM).
- **Phần mềm:** Python 3.9+, Flask, Scikit-learn, Underthesea.

## 4.2. Đánh giá và so sánh các mô hình Machine Learning

Hệ thống đã chạy thử nghiệm tập Test (3,166 bình luận) qua 4 mô hình ML và đánh giá qua các chỉ số chuẩn: Accuracy (Độ chính xác tổng thể), Precision, Recall và F1-Score.

| Mô hình (Model) | Accuracy | Precision | Recall | F1-Score | Thời gian Train | Thời gian Predict |
|-----------------|----------|-----------|--------|----------|-----------------|-------------------|
| Naive Bayes | 86.89% | 0.8799 | 0.8689 | 0.8471 | ~0.01s | ~0.001s |
| Logistic Regression | 89.36% | 0.8845 | 0.8936 | 0.8760 | ~0.22s | ~0.001s |
| **Support Vector Machine (SVM)**| **89.39%** | **0.8847** | **0.8939** | **0.8789** | ~8.28s | ~1.168s |
| Random Forest | 87.97% | 0.8713 | 0.8797 | 0.8701 | ~1.54s | ~0.041s |

**Đánh giá:**
- Mô hình **SVM** đạt hiệu năng cao nhất (89.39%) trong phân loại sắc thái bình luận. Mặc dù thời gian huấn luyện lâu hơn một chút (8 giây), nhưng kết quả suy luận trên hơn 3000 mẫu chỉ tốn khoảng 1 giây, hoàn toàn đáp ứng được tính thời gian thực.
- Logistic Regression bám sát nút với độ chính xác 89.36% và tốc độ huấn luyện/dự đoán cực kỳ ấn tượng.

*(Lưu ý: Các biểu đồ trực quan So sánh Độ chính xác, Thời gian huấn luyện và Confusion Matrix đã được hệ thống xuất tự động lưu trong thư mục `results/`)*.

## 4.3. So sánh Machine Learning (SVM) và Google Gemini API

Để có cái nhìn khách quan, nhóm tiến hành đánh giá sức mạnh giữa phương pháp truyền thống (SVM) và AI tạo sinh hiện đại (Gemini API). Do giới hạn Free-tier của Gemini (5 request/phút), việc đánh giá API được lấy trên một tập mẫu giả lập/thu nhỏ (100 samples) nhưng vẫn phản ánh đúng bản chất.

| Tiêu chí | SVM (Machine Learning) | Google Gemini 2.5 Flash |
|----------|------------------------|-------------------------|
| **Độ chính xác (Accuracy)** | ~89.4% (Trên tập test 3000 mẫu) | ~88.0% |
| **Tốc độ phản hồi** | Cực nhanh (< 0.01 giây/câu) | Chậm (1-3 giây/câu, phụ thuộc mạng) |
| **Phân tích hàng loạt (Batch)** | Rất mạnh mẽ (Hàng ngàn dòng/giây) | Yếu (Bị giới hạn Rate limit quota) |
| **Internet & Chi phí** | Offline, Miễn phí hoàn toàn | Cần Internet, Tốn phí nếu vượt quota |
| **Giải thích kết quả** | Dựa trên độ tin cậy của thuật toán | Có thể đưa ra văn bản giải thích lý do cụ thể |
| **Tùy chỉnh lĩnh vực** | Tốt (Học từ Dataset cụ thể của bài toán) | Phụ thuộc vào kiến thức nền của LLM |

## 4.4. Demo thực tế trên Giao diện Web

Ứng dụng web hoạt động ổn định và đáp ứng mượt mà cả hai phương thức.

**Bình luận tích cực:**
- "Sản phẩm rất tốt, đóng gói cẩn thận, giao hàng nhanh." → **Positive** (SVM tự tin >90%).

**Bình luận trung tính:**
- "Sản phẩm có 3 màu: đen, trắng, xanh" → **Neutral**.

**Bình luận tiêu cực:**
- "Chất lượng kém, giao trễ 1 tuần không gọi điện" → **Negative**.

**Lưu ý khi sử dụng thực tế:**
- Khi người dùng upload một file CSV có 1000 bình luận: Nếu dùng Gemini, hệ thống sẽ mất hàng giờ và liên tục báo lỗi Quota (giới hạn API). Tuy nhiên, nếu chuyển Dropdown sang **Machine Learning**, hệ thống sẽ hoàn tất và lưu database toàn bộ 1000 bình luận chỉ trong chớp mắt.

---

# 5. Kết luận

## 5.1. Tổng kết

Trong báo cáo này, nhóm đã trình bày quá trình xây dựng và triển khai thành công **Hệ thống phân tích cảm xúc bình luận** áp dụng cả hai phương pháp: Trí tuệ nhân tạo tạo sinh (Google Gemini) và Học máy truyền thống (Scikit-Learn). 

Các kết quả chính đạt được:
1. Xử lý chuẩn xác bộ dữ liệu tiếng Việt UIT-VSFC.
2. Xây dựng và so sánh hiệu suất 4 thuật toán ML. Chọn ra SVM làm mô hình cốt lõi với độ chính xác xấp xỉ 90%.
3. Phát triển hệ thống Web App (Flask) hoàn thiện có khả năng chạy nền (Background processing) hỗ trợ xử lý hàng ngàn dữ liệu.

## 5.2. Đánh giá ưu và nhược điểm

### 5.2.1. Ưu điểm
- **Linh hoạt:** Cung cấp cho người dùng 2 lựa chọn (Gemini để có giải thích sâu sắc, hoặc SVM để xử lý nhanh hàng loạt).
- **Ổn định cao:** Khắc phục hoàn toàn bài toán đứt cáp mạng, nghẽn server hoặc hết tiền API nhờ có mô hình ML chạy cục bộ (offline).
- **Giao diện trực quan:** Bảng điều khiển dễ dùng, thể hiện thống kê rõ ràng.

### 5.2.2. Nhược điểm
- Mô hình ML hiện tại (TF-IDF + SVM) chưa xử lý triệt để các bình luận mang tính mỉa mai, châm biếm (sarcasm) tốt như mô hình LLM.
- Chữ viết tắt, teencode phức tạp (chưa có trong dữ liệu huấn luyện) có thể làm giảm độ tin cậy của SVM.

## 5.3. Hướng phát triển trong tương lai

1. **Ứng dụng Deep Learning:** Thay thế TF-IDF + SVM bằng các mô hình học sâu như PhoBERT (dành riêng cho tiếng Việt) để cải thiện khả năng hiểu ngữ cảnh dài.
2. **Xây dựng từ điển Teencode:** Tạo bộ ánh xạ chuẩn hóa từ vựng giới trẻ trước khi đưa vào mô hình để tăng độ chính xác.
3. **Mở rộng biểu đồ thống kê:** Tích hợp trực tiếp các biểu đồ Word Cloud (đám mây từ vựng) lên giao diện Web để khách hàng thấy được những từ khóa nào đang được nhắc đến nhiều nhất.

---

# TÀI LIỆU THAM KHẢO

1. Kieu, B. T., & Nguyen, N. L. T. (2018). *Vietnamese Students’ Feedback Corpus (UIT-VSFC)*.
2. Google. (2024). *Google Gemini API Documentation*. https://ai.google.dev/
3. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830.
4. Flask Documentation. (2024). *Flask - Web Development, One Drop at a Time*. https://flask.palletsprojects.com/
5. Underthesea. (2024). *Vietnamese NLP Toolkit*. https://github.com/undertheseanlp/underthesea

---

Ngày hoàn thành: [Ngày/tháng/năm]

Điểm đánh giá: ___________

Nhận xét của giảng viên:

_______________________________________________

_______________________________________________

Chữ ký giảng viên: ____________________
