
# app_streamlit.py
# ------------------------------------------------------------
# Giao diện web 1-file viết bằng Streamlit cho hệ thống gợi ý.
# Mục tiêu: chạy thẳng bằng Python, không cần HTML/CSS/JS riêng.
# Cách chạy:
#   pip install streamlit pandas torch sentence-transformers torch-geometric torchvision
#   streamlit run app_streamlit.py
# Lưu ý:
#   - Thuật toán "multi-modal" cần các thư viện nặng (torch, torchvision,
#     sentence-transformers, torch-geometric). Nếu thiếu, app vẫn chạy
#     các thuật toán còn lại (collaborative / content-based / hybrid).
#   - File này giả định bạn có các CSV: users_expanded.csv, products_expanded.csv,
#     purchases_expanded.csv, browsing_history_expanded.csv, product_images_expanded.csv
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import logging
import os

# Cố gắng import các thành phần của mô hình. Nếu thiếu phụ thuộc nặng,
# ta vẫn cho phép chạy các thuật toán đơn giản.
multimodal_ok = True
try:
    import torch
    from model import (
        collaborative_filtering,     # Gợi ý dựa trên người dùng tương tự (CF)
        content_based_filtering,     # Gợi ý dựa trên nội dung/sản phẩm tương tự
        hybrid_recommendation,       # Kết hợp CF + content-based
        MultiModalModel              # Mô hình đa phương thức (ID + ảnh + text + đồ thị)
    )
except Exception as e:
    multimodal_ok = False
    # Nếu không import được đầy đủ, vẫn thử import các hàm cơ bản
    try:
        from model import collaborative_filtering, content_based_filtering, hybrid_recommendation
    except Exception:
        # Nếu cả các hàm cơ bản cũng không import được thì dừng hẳn
        raise

# Cấu hình logging mức INFO để xem các thông báo trong terminal khi chạy.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("streamlit-app")

# Thiết lập tiêu đề và layout trang Streamlit
st.set_page_config(page_title="Product Recommender", layout="wide")
st.title("Product Recommender — Streamlit UI (1-file)")

# Hàm tiện ích: nạp CSV có cache để tăng tốc (không đọc lại nhiều lần)
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    """
    Nạp một file CSV. Nếu không tồn tại, trả về DataFrame rỗng và cảnh báo.
    Dùng @st.cache_data để cache kết quả theo nội dung file.
    """
    if not os.path.exists(path):
        st.warning(f"Không tìm thấy file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)

# Đọc dữ liệu đầu vào cho hệ thống gợi ý
users = load_csv('users_expanded.csv')                   # Danh sách user và thuộc tính
products = load_csv('products_expanded.csv')             # Danh mục sản phẩm (id, tên, mô tả, rating,...)
product_images = load_csv('product_images_expanded.csv') # Mapping product_id -> đường dẫn ảnh, view
purchases = load_csv('purchases_expanded.csv')           # Lịch sử mua
browsing_history = load_csv('browsing_history_expanded.csv') # Lịch sử xem/browse

# --------- SIDEBAR: các điều khiển chính cho người dùng ---------
st.sidebar.header("Thiết lập")
# Nếu thiếu file users, không thể xác định user_id
if users.empty:
    st.error("Thiếu file users_expanded.csv — không thể chạy demo.")
    st.stop()

# Lấy danh sách user_id để gợi ý giá trị mặc định
user_ids = users['user_id'].dropna().astype(int).sort_values().tolist()
default_user = user_ids[0] if user_ids else 1

# number_input: chọn User ID để tạo gợi ý
user_id = st.sidebar.number_input("User ID", min_value=1, value=default_user, step=1)

# Chọn thuật toán gợi ý
algorithms = ["collaborative", "content-based", "hybrid"]
if multimodal_ok:
    # Chỉ cho phép chọn "multi-modal" khi import đủ phụ thuộc
    algorithms.append("multi-modal")
algorithm = st.sidebar.selectbox("Thuật toán", algorithms, index=0)

# Số lượng gợi ý tối đa cần hiển thị
top_k = st.sidebar.slider("Số gợi ý tối đa", 1, 50, 10)

# --------- HIỂN THỊ DỮ LIỆU DANH MỤC ---------
with st.expander("Danh mục sản phẩm (products_expanded.csv)"):
    # Cho phép người dùng xem nhanh danh sách sản phẩm gốc
    st.dataframe(products)

# Kiểm tra tính hợp lệ của user_id được nhập
if user_id not in users['user_id'].values:
    st.error("User ID không tồn tại trong dữ liệu.")
    st.stop()


