# Movie Recommendation System

A comprehensive movie recommendation system implementing various collaborative filtering and content-based approaches.

## English

### Project Overview
This project implements multiple recommendation algorithms including:
- User-based Collaborative Filtering
- Item-based Collaborative Filtering
- Matrix Factorization
- Content-based Filtering
- Hybrid Approaches

### Requirements
- Python 3.8+
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib

### Installation
1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Data Setup
1. Download the MovieLens 100K dataset
2. Place the dataset in the `data/ml-100k/` directory with the following structure:
```
data/
  ml-100k/
    u.user
    u.item
    ua.base
    ua.test
```

### Running the Code
1. Content-based Filtering:
```bash
python test.py
```

2. Matrix Factorization:
```bash
python matrixFactorization.py
```

3. User-based Collaborative Filtering:
```bash
python userKNN.py
```

4. Item-based Collaborative Filtering:
```bash
python itemKNN.py
```

5. Hybrid Approach:
```bash
python hybrid.py
```

### Results
- Evaluation results are saved in the `results/` directory
- Visualization plots are saved in the `plots/` directory

## Tiếng Việt

### Tổng Quan Dự Án
Dự án này triển khai nhiều thuật toán gợi ý khác nhau bao gồm:
- Lọc cộng tác dựa trên người dùng
- Lọc cộng tác dựa trên sản phẩm
- Phân tích ma trận
- Lọc dựa trên nội dung
- Các phương pháp kết hợp

### Yêu Cầu
- Python 3.8+
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib

### Cài Đặt
1. Clone repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Cài đặt các gói cần thiết:
```bash
pip install -r requirements.txt
```

### Thiết Lập Dữ Liệu
1. Tải xuống bộ dữ liệu MovieLens 100K
2. Đặt dữ liệu vào thư mục `data/ml-100k/` với cấu trúc sau:
```
data/
  ml-100k/
    u.user
    u.item
    ua.base
    ua.test
```

### Chạy Code
1. Lọc dựa trên nội dung:
```bash
python test.py
```

2. Phân tích ma trận:
```bash
python matrixFactorization.py
```

3. Lọc cộng tác dựa trên người dùng:
```bash
python userKNN.py
```

4. Lọc cộng tác dựa trên sản phẩm:
```bash
python itemKNN.py
```

5. Phương pháp kết hợp:
```bash
python hybrid.py
```

### Kết Quả
- Kết quả đánh giá được lưu trong thư mục `results/`
- Các biểu đồ trực quan được lưu trong thư mục `plots/`