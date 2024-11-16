# Chọn image cơ sở từ Python 3.10
FROM python:3.10-slim

# Cài đặt các công cụ cần thiết để build thư viện C (như lightfm)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libc-dev \
    liblapack-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Đặt thư mục làm việc trong container
WORKDIR /app

# Sao chép các file cần thiết vào container
COPY . /app/

# Cài đặt các thư viện trong requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Mở cổng cho ứng dụng (nếu ứng dụng của bạn chạy trên một cổng cụ thể, ví dụ Flask chạy trên cổng 5000)
EXPOSE 5000

# Chạy ứng dụng của bạn (ví dụ sử dụng app.py nếu là Flask app)
CMD ["python", "app.py"]
