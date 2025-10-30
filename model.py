import logging
import pandas as pd
# Hiển thị tất cả log từ mức DEBUG trở lên
logging.basicConfig(level=logging.DEBUG)
# tạo logger riêng cho module
logger = logging.getLogger(__name__)

'''Hàm gợi ý dựa trên cộng tác với:
    - user_id là người dùng đang được gợi ý
    - purchases là dataframe lịch sử mua sắm của tất cả người dùng
    - products là dataframe mô tả sản phẩm'''
def collaborative_filtering(user_id: int, purchases: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    # ghi trong file lod=g để cho biết hàm đang chạy cho user nào
    logger.debug(f"Collaborative Filtering for user_id: {user_id}")
    # lấy cột product id và mà user id = user id đang xét, lấy các product id ko trùng lặp
    # B1: lấy danh sách sản phẩm của người dùng hiện tại
    user_purchases = purchases[purchases['user_id'] == user_id]['product_id'].unique()
    # ghi ra danh sách sản phẩm dưới dạng series
    logger.debug(f"User purchases: {user_purchases}")
    # lấy những cột user_id mà product id nằm trong user_purchases và user id khác người dùng hiện tại
    # B2: tìm những người mua cùng sản phẩm với người dùng đang xét
    other_users = purchases[purchases['product_id'].isin(user_purchases) & purchases['user_id'] != user_id]['user_id'].unique()
    # B3: lấy danh sách sản phẩm của những người dùng khác
    other_purchases = purchases[purchases['user_id'].isin(other_users)]
    # B4: đếm số lần xuất hiện của các sản phẩm trong other_purchases
    product_counts = other_purchases['product_id'].value_counts()
    # B5: chọn danh sách sản phẩm gợi ý
    # lấy những sản phẩm ở trong product count (danh sách mua của người dùng khác) mà ko nằm trong ds mua của người dùng đang xét
    recommendations = products[products['product_id'].isin(product_counts.index) &
                               ~products['product_id'].isin(user_purchases)].copy()
    # tính điểm
    # xét các product id trong bảng recommendations, tìm product id giống thế trong product count, gắn giá trị đếm tương ứng
    # nếu ko tìm thấy thì ghi 0 vào cột mới purchase_count
    recommendations['purchase_count'] = recommendations['product_id'].map(product_counts).fillna(0)
    # tính điểm dựa trên số lần xuất hiện * đánh giá
    recommendations['raw_score'] = recommendations['purchases_count']*recommendations['rating']
    # chuẩn hóa về thang [0,1] bằng cách chia cho lần xuất hiện nhiều nhất
    recommendations['score'] = recommendations['raw_score']/product_counts.max()
    # gắn nhãn nguồn
    recommendations['source'] = 'Collaborative Filtering'
    # ghi lại dataframe recommendations với 3 cột  product_id, score, source vào log
    logger.debug(f"Collaborative recommendations: \n{recommendations[['product_id', 'score', 'source']]}")
    
    return recommendations.sort_values(by='score',ascending = False)

'''Hàm gợi ý dựa trên lịch sử xem với:
    - user_id: người dùng đang được gợi ý
    - purchases: dataframe ghi lịch sử mua
    - browsing_history: dataframe ghi lịch sử xem sản phẩm
    - products: dataframe mô tả sản phẩm'''
def content_based_filtering(user_id: int, purchases: pd.DataFrame, browsing_history: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    logger.debug(f"Content-Based Filtering for user_id: {user_id}")
    # lấy những sản phẩm mà người dùng đang xét đã xem
    user_history = browsing_history[browsing_history['user_id'] == user_id]['product_id'].unique()
    # ghi ra danh sách sản phẩm 
    logger.debug(f"User browsing history: {user_history}")
    # lấy thông tin của những sản phẩm mà người dùng đã xem
    user_products = products[products['product_id'].isin(user_history)]
    # nếu như sản phẩn có thông tin và cột category nằm trong dataframe products
    if not user_products.empty and 'category' in products.columns:
        # gợi ý những sản phẩm mà có category nằm trong user_products mà không phải là những sản phẩm mà người dùng đã xem
        recommendations = products[products['category'].isin(user_products['category']) & 
                                   ~products['product_id'].isin(user_history)].copy()
        
        # lấy trung bình rating các sản phẩm mà người dùng đã xem 
        avg_rating = user_products['rating'].mean()
        # chuẩn hóa rating về thang [0,1]
        recommendations['score'] = recommendations['rating']/5.0 * avg_rating
        recommendations['score'] = recommendations['score'] / recommendations['score'].max()
    else:
        # trả về dataframe rỗng chỉ có tên cột
        recommendations = pd.DataFrame(columns = ['product_id', 'product_name', 'price', 'rating', 'score', 'source'])
        # ghi lại trong log là không có gợi ý theo nội dung
        logger.debug("No content-based recommendations.")
    recommendations['source'] = 'Content-Based Filtering'
    logger.debug(f"Content-based recommendations:\n{recommendations[['product_id', 'score', 'source']]}")
    return recommendations

def hybrid_recommendation(user_id, purchases, browsing_history, products):
    logger.debug(f"Hybrid Recommendation for user_ id: {user_id}")
    # danh sách sản phẩm mua 
    user_purchases = purchases[purchases['user_id'] == user_id]['product_id'].unique()
    # danh sách sản phẩm đã xem
    user_browsed = browsing_history[browsing_history['user_id'] == user_id]['product_id'].unique()
    # tổng hợp danh sách đã mua và đã xem
    user_history = set(user_purchases).union(user_browsed)
    logger.debug(f"User history (purchases + browsed): {user_history}")
    # lấy danh sách gợi ý của 2 hàm gợi ý
    collab_recs = collaborative_filtering(user_id, purchases, products)
    content_recs = content_based_filtering(user_id, purchases, browsing_history, products)
    # ghép 2 dataframe lại thành 1 danh sách gợi ý tổng
    all_recommendations = pd.concat([collab_recs, content_recs], ignore_index=True)
    logger.debug(f"Combined recommendations:\n{all_recommendations[['product_id', 'score', 'source']]}")
    # nếu không có sản phẩm gợi ý nào
    if all_recommendations.empty:
        logger.debug("No recommendations; adding popular products.")
        # gợi ý những sản phẩm được mua nhiều nhất
        popular_products = purchases['product_id'].value_counts().head(3).index
        all_recommendations = products[products['product_id'].isin(popular_products) & 
                                      ~products['product_id'].isin(user_history)].copy()
        # đặt điểm của các sản phẩm đó là 0.5
        all_recommendations['score'] = 0.5
        all_recommendations['source'] = 'Popular Products'
    # gợi ý cuối cùng là sắp xếp all_recommendations thep thứ tự giảm dần của score, loại bỏ những sp bị lặp, chỉ giữ cái đầu tiên
    final_recommendations = all_recommendations.sort_values(by='score', ascending=False) \
                                               .drop_duplicates(subset=['product_id'], keep='first')  
    logger.debug(f"Final hybrid recommendations:\n{final_recommendations[['product_id', 'score', 'source']]}")
    
    return final_recommendations