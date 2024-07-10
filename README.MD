Đây là một tài liệu README hướng dẫn cài đặt chương trình website Flask:

---

# Hướng dẫn cài đặt chương trình website Flask

Chương trình này được viết bằng Python và sử dụng framework Flask để xây dựng một ứng dụng web đơn giản.

## Yêu cầu

1. Python 3 đã được cài đặt. Bạn có thể tải Python từ [trang web chính thức của Python](https://www.python.org/downloads/).

## Cài đặt

1. Tải mã nguồn từ repository này hoặc sao chép mã nguồn vào máy tính của bạn.
2. Mở Command Prompt (cmd) hoặc Terminal và di chuyển đến thư mục chứa mã nguồn của chương trình.
3. Tạo một môi trường ảo (virtual environment) bằng cách chạy lệnh sau:

    ```bash
    python -m venv venv
    ```

4. Kích hoạt môi trường ảo bằng lệnh:

    Trên Windows:

    ```bash
    venv\Scripts\activate
    ```

    Trên Linux/macOS:

    ```bash
    source venv/bin/activate
    ```

5. Cài đặt các gói cần thiết bằng lệnh:

    ```bash
    pip install -r requirements.txt
    ```

6. Khởi động ứng dụng bằng lệnh:

    ```bash
    python app.py
    ```

7. Truy cập `http://127.0.0.1:5000/` trong trình duyệt của bạn để sử dụng chương trình.

## Mô tả

-   Chương trình này là một ứng dụng web đơn giản được viết bằng Flask.
-   Nó cung cấp chức năng dự đoán khả năng mắc bệnh tiểu đường dựa trên các thông số y tế cơ bản.

## Cấu trúc thư mục

-   `app.py`: File chính của ứng dụng, bao gồm các route và logic xử lý.
-   `templates/index.html`: File HTML cho giao diện người dùng.
-   `models/random_forest_model.pkl`: File chứa mô hình học máy được sử dụng để dự đoán.

---

Vui lòng làm theo hướng dẫn trên để cài đặt và chạy chương trình thành công. Nếu gặp bất kỳ vấn đề nào, đừng ngần ngại liên hệ với tôi.

Chúc bạn thành công! 🚀

---

Hãy nhớ thay đổi phần mô tả, cấu trúc thư mục và các yêu cầu tùy thuộc vào dự án cụ thể của bạn.
