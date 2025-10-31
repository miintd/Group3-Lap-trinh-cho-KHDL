# ------------------ IMPORTS (Dòng ~16–20) ------------------
# Flask: web framework; render_template/request/flash/redirect/url_for cho flow web
# pandas: đọc/ xử lý CSV -> DataFrame
# torch: chạy model PyTorch (multi-modal)
# import từ model.py: các hàm/mô hình gợi ý dùng trong app
from flask import Flask, render_template, request, flash, redirect, url_for
import pandas as pd
import torch
from model import collaborative_filtering, content_based_filtering, hybrid_recommendation, MultiModalModel
import logging

# ------------------ APP + LOGGER (Dòng ~22–26) ------------------
app = Flask(__name__)              # khởi app Flask
app.secret_key = 'your-secret-key' # cần cho flash/session (dev only)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)  # logger cho file này

# ------------------ LOAD DATA + MODEL (Dòng ~29–37) ------------------
# đọc CSV (users, products, ảnh, purchases, browsing)
users = pd.read_csv('users_expanded.csv')
products = pd.read_csv('products_expanded.csv')
product_images = pd.read_csv('product_images_expanded.csv')
purchases = pd.read_csv('purchases_expanded.csv')
browsing_history = pd.read_csv('browsing_history_expanded.csv')

# khởi model multi-modal với kích thước dựa trên số user/product
num_users = users['user_id'].nunique()
num_products = products['product_id'].nunique()
model = MultiModalModel(num_users, num_products)

# ------------------ ROUTE: index (Dòng ~39–45) ------------------
@app.route('/')
def index():
    # hiển thị trang chủ, truyền danh sách products (list of dict) sang template
    return render_template('index.html', products=products.to_dict(orient='records'))

# ------------------ ROUTE: /recommend (Dòng ~47–82) ------------------
@app.route('/recommend', methods=['POST'])
def get_recommendations():
    try:
        # nhận form: user_id, algorithm
        user_id = int(request.form['user_id'])
        algorithm = request.form['algorithm']
        logger.debug(f"Processing request for user_id: {user_id}, algorithm: {algorithm}")

        # kiểm tra user tồn tại
        if user_id not in users['user_id'].values:
            flash('User ID not found!')
            return redirect(url_for('index'))

        # LẤY lịch sử tương tác (mua + xem)
        purchased_product_ids = purchases[purchases['user_id'] == user_id]['product_id'].unique()
        browsed_product_ids = browsing_history[browsing_history['user_id'] == user_id]['product_id'].unique()

        # tập các sản phẩm user đã tương tác, thêm cột nguồn (Purchased/Browsed)
        interacted_products = products[products['product_id'].isin(purchased_product_ids) |
                                      products['product_id'].isin(browsed_product_ids)].copy()
        interacted_products['source'] = interacted_products['product_id'].apply(
            lambda x: 'Purchased' if x in purchased_product_ids else 'Browsed'
        )
        logger.debug(f"Interacted products: {interacted_products['product_id'].tolist()}")

        # CHỌN thuật toán tương ứng để sinh recommendations
        if algorithm == 'collaborative':
            # dựa vào hành vi người dùng khác
            recommendations = collaborative_filtering(user_id, purchases, products)
        elif algorithm == 'content-based':
            # dựa vào đặc trưng sản phẩm / mô tả
            recommendations = content_based_filtering(user_id, purchases, browsing_history, products)
        elif algorithm == 'hybrid':
            # kết hợp collaborative + content-based
            recommendations = hybrid_recommendation(user_id, purchases, browsing_history, products)
        elif algorithm == 'multi-modal':
            # dùng model PyTorch: truyền user, product ids, texts, images -> lấy score
            product_ids = torch.LongTensor(products['product_id'].values) - 1  # 0-index cho embedding
            texts = products['description'].tolist()
            with torch.no_grad():
                outputs = model(
                    torch.LongTensor([user_id - 1]),
                    product_ids,
                    texts,
                    edge_index=None,
                    product_images_df=product_images
                )
            # chuyển embedding -> điểm (hiện tại dùng mean)
            scores = outputs.mean(dim=1).cpu().numpy()
            recommendations = products.copy()
            recommendations['score'] = scores
            recommendations['source'] = 'Multi-Modal'
        else:
            flash('Invalid algorithm selected!')
            return redirect(url_for('index'))

        # LỌC bỏ sản phẩm user đã xem/mua (không gợi lại)
        recommended_products = recommendations[~recommendations['product_id'].isin(purchased_product_ids) &
                                               ~recommendations['product_id'].isin(browsed_product_ids)].copy()
        logger.debug(f"Filtered recommendations:\n{recommended_products[['product_id', 'score', 'source']]}")

        # nếu không còn sản phẩm phù hợp -> thông báo
        if recommended_products.empty:
            flash('No recommendations available for this user.')

        # trả template chứa interacted + recommended
        return render_template('recommendations.html',
                               interacted_products=interacted_products.to_dict(orient='records'),
                               recommended_products=recommended_products.to_dict(orient='records'))
    except Exception as e:
        # bắt lỗi chung: log + flash + redirect về index
        logger.error(f"Error in get_recommendations: {str(e)}")
        flash(f'An error occurred: {str(e)}')
        return redirect(url_for('index'))

# ------------------ RUN (Cuối file) ------------------
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    # debug=True cho dev (auto reload). Production nên dùng Gunicorn/uWSGI.
    app.run(host='0.0.0.0', port=port, debug=True)
