# Human Detector - YOLOv8

Dự án này tập trung vào việc xây dựng và huấn luyện mô hình phát hiện người (Human Detection) sử dụng kiến trúc YOLOv8 từ thư viện Ultralytics, kèm theo giao diện web để demo.

## Tính năng chính
- **Huấn luyện mô hình:** Quy trình huấn luyện tự động với YOLOv8s trên tập dữ liệu người.
- **Đánh giá (Validation):** Kiểm tra hiệu năng mô hình với các chỉ số mAP.
- **Dự đoán (Inference):** Hỗ trợ nhận diện người từ hình ảnh local, URL và video YouTube.
- **Xuất mô hình:** Chuyển đổi sang định dạng ONNX để triển khai đa nền tảng.
- **Web App Demo:** Giao diện web với Streamlit để demo real-time detection.

## Cấu trúc dự án
- `YOLOv8_Training.ipynb`: Notebook chứa toàn bộ mã nguồn huấn luyện và thử nghiệm.
- `app.py`: Streamlit web app chính.
- `config.py`: File cấu hình ứng dụng.
- `utils.py`: Các hàm tiện ích xử lý ảnh và detection.
- `requirements.txt`: Các dependencies cần thiết.
- `README.md`: Hướng dẫn và tóm tắt dự án.

##  Quy trình thực hiện (tóm tắt từ Notebook)
1. **Cài đặt môi trường:** Cài đặt thư viện `ultralytics` và các phụ thuộc.
2. **Dữ liệu:** Tải và giải nén bộ dữ liệu `human_detection_dataset`.
3. **Huấn luyện:** Sử dụng mô hình `yolov8s.pt` huấn luyện trong 20 epochs, kích thước ảnh 640.
4. **Kiểm thử:** Đánh giá độ chính xác trên tập val.
5. **Dự đoán:** Thử nghiệm nhận diện trên các nguồn ảnh/video thực tế.
6. **Xuất bản:** Lưu trữ mô hình dưới dạng `.pt` và `.onnx`.

## Cài đặt và chạy Web App

### Yêu cầu
- Python 3.8+
- pip

### Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Chạy Web App
```bash
streamlit run app.py
```

Web app sẽ mở tại `http://localhost:8501`

### Tính năng Web App
- Upload ảnh để detect người
- Điều chỉnh confidence threshold
- Xem kết quả detection real-time
- Download ảnh đã xử lý
- Giao diện thân thiện, dễ sử dụng

## Kết quả huấn luyện (Tham khảo)
Dựa trên kết quả chạy trong notebook:
- **mAP50:** ~0.847
- **mAP50-95:** ~0.598

## Lộ trình phát triển
- [x] Web app cơ bản với Streamlit
- [ ] Video processing
- [ ] Real-time webcam detection
- [ ] People counting
- [ ] Tracking (DeepSORT, ByteTrack)
- [ ] Heatmap visualization
- [ ] Docker deployment
- [ ] API endpoint

---
*Dự án được thực hiện nhằm mục đích nghiên cứu và ứng dụng Computer Vision trong nhận diện đối tượng.*
