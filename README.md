# 🏪 Smart Store People Counting System (YOLOv8)

## 📺 Demo
<div align="center">
  <video src="outputs\videos\counted_clip9_1780662859.mp4" width="800" controls autoplay loop muted></video>
</div>

Dự án xây dựng hệ thống đếm số lượng người ra vào cửa hàng/siêu thị thông minh sử dụng thị giác máy tính. Hệ thống kết hợp mô hình YOLOv8 để nhận diện người và các thuật toán theo dõi đối tượng (Multi-Object Tracking) để đếm lượt băng qua vạch (Line Crossing).

## 🚀 Tính năng chính
- **Phát hiện người (Human Detection):** Sử dụng YOLOv8 (với trọng số đã được huấn luyện tối ưu).
- **Theo dõi đối tượng (Object Tracking):** Hỗ trợ ByteTrack và BoTSort giúp bám đuổi ID người ổn định qua các khung hình.
- **Đếm người theo vạch (Line Crossing):** Tự động đếm số người đi VÀO (IN) và đi RA (OUT) dựa trên vạch cấu hình linh hoạt (ngang/dọc).
- **Cảnh báo mật độ (Crowd Alert):** Thông báo khi số lượng người hiện có trong khu vực vượt ngưỡng cho phép.
- **Web App Demo (Streamlit):** Giao diện web trực quan, cho phép upload video, cấu hình vạch đếm trực tiếp và tải kết quả.
- **Xuất báo cáo:** Tự động tạo file CSV ghi lại lịch sử các sự kiện băng qua vạch với mốc thời gian chi tiết.

## 📁 Cấu trúc dự án
- `app/app.py`: Giao diện Web chính (Streamlit).
- `app/counter.py`: Logic cốt lõi xử lý việc đếm người và theo dõi ID.
- `app/utils.py`: Các hàm bổ trợ xử lý hình ảnh và video.
- `scripts/process_video.py`: Script CLI để xử lý video offline quy mô lớn.
- `scripts/train.py` & `predict.py`: Công cụ huấn luyện và dự đoán cơ bản trên ảnh.
- `models/`: Chứa các file trọng số (`best.pt`, `best.onnx`).
- `outputs/`: Thư mục lưu trữ video kết quả và logs.
- `sample_videos/`: Video mẫu để kiểm tra hệ thống.

## 🛠️ Cài đặt

### 1. Yêu cầu hệ thống
- Python 3.8 trở lên.
- Nên sử dụng GPU (NVIDIA) để đạt hiệu năng xử lý thời gian thực tốt nhất.

### 2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

## 💻 Hướng dẫn sử dụng

### 1. Chạy Giao diện Web (Khuyên dùng)
Giao diện này cho phép bạn cấu hình vạch đếm (Line position) một cách trực quan.
```bash
streamlit run app/app.py
```
**Cách sử dụng trên Web:**
1. Chọn file video đầu vào.
2. Điều chỉnh **Line Orientation** (Ngang/Dọc) và **Line Position** trên thanh sidebar để khớp với vị trí cửa ra vào.
3. Thiết lập **Crowd Threshold** để nhận cảnh báo mật độ.
4. Nhấn **Process Video** và đợi hệ thống xử lý.
5. Xem kết quả trực tiếp hoặc tải xuống video đã được gán nhãn và file CSV báo cáo.

### 2. Sử dụng dòng lệnh (CLI)
Dành cho việc xử lý hàng loạt hoặc tích hợp vào hệ thống khác:
```bash
python scripts/process_video.py --source "path/to/video.mp4" --output "outputs/result.mp4" --line-position 400 --save-csv
```

## 📊 Kết quả huấn luyện
Hệ thống sử dụng mô hình YOLOv8 Small được tinh chỉnh:
- **mAP50:** ~0.847
- **mAP50-95:** ~0.598
- Hiệu năng: Xử lý mượt mà trên các dòng GPU phổ thông.

## 🗺️ Lộ trình phát triển (Roadmap)
- [x] Web app tương tác với Streamlit.
- [x] Video processing & Object Tracking.
- [x] Đếm người IN/OUT qua vạch.
- [x] Xuất báo cáo CSV sự kiện.
- [ ] Tích hợp Webcam/Camera RTSP trực tiếp.
- [ ] Hiển thị biểu đồ thống kê (Heatmap, Time-series).
- [ ] Đóng gói Docker để triển khai nhanh.

---
*Dự án được phát triển nhằm mục đích cung cấp giải pháp phân tích dữ liệu khách hàng cho các cửa hàng bán lẻ.*
