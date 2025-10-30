
import pandas as pd
import logging
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from sentence_transformers import SentenceTransformer
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import os

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MultiModalModel(nn.Module):
        # -Hàm này có tác dụng là tạo embedding vector cho các loại thông tin như ID Khách hàng , ID Sản Phẩm
        #-Tạo vector embedding cho các loại hình ảnh, tạo vector embedding để phân loại các loại phong cách rồi theo phong 
        #cách đánh giá lại hình ảnh theo tiêu chí của phong cách , Xử lý theo model đã có sãn là Resnet50
        #-Tạo vector embedding cho đoạn văn , chữ cái theo model đã được trained sẵn đó là SentenceTransformer 
        #-Tạo vector embedding chứa cả ba loại vector embedding trên
        #Tạo một vector embedding cho phép dùng các truy cập vào các thông tin của các vector embedding (model xung quanh)  

    def __init__(self, num_users, num_products, embedding_dim=128):
        super().__init__() # Hàm này có tác dụng là khai báo để class của mình có thể sử dụng các chức năng của nn.model trong pytorch
        # hàm này là kế thừa các các cái biến nằm trong nn.module

        # Tạo một lớp (layer) mà trong đó chứa các vector embedding của tất cả các user ID có thể có trong bảng 
        self.user_emb = nn.Embedding(num_users, embedding_dim)

        #Tạo một lớp (layer) mà trong đó chứa vector của produt ID có thể có trong bảng 
        self.product_emb = nn.Embedding(num_products, embedding_dim)
        
        # Tạo một layer xử lý hình ảnh mà có tất cả các dữ liệu của resnet50 đã có sẵn mình chỉ lại chuyển đổi tên thôi 
        self.image_encoder = resnet50(pretrained=True)

        # Thay đổi đầu ra của resnet50, ban đầu là size vector là 1000 chuyển thành 128 
        self.image_encoder.fc = nn.Linear(2048, embedding_dim)
        
        # Tạo một lớp (layer) sao cho chứa bốn góc nhìn của sản phẩm mặt trước, mặt hông, mặt sau, và toàn thân 
        # Dùng để phân biệt ảnh của sản phâm trong các góc nhìn khác nhau 
        self.view_embedding = nn.Embedding(4, embedding_dim)  # front, side, back, full

        # tạo một layer chuyển các ảnh từ các góc nhìn , từ các ảnh ban đầu thành một vector mới thể hiện theo phong cách 
        self.style_projection = nn.Linear(embedding_dim, embedding_dim)
        

        for name, param in self.image_encoder.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        # Tải thư viện pretrained có sẵn là all-MiniLM-L6-v2 rồi đặt tên của lại cho cái lớp này thành Chuyen_Hoa_Chu_Doan_Van
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')

         # Đặt mặc định của kết quả của vector embedding thành 128 chiều 
        self.text_proj = nn.Linear(384, embedding_dim)
        
        # Lệnh này cho phép chúng ta truy cập thông tin của các node (embedding vector) xung quanh của nó để đánh giá tính chất của vector hiện tại 
        self.conv1 = GCNConv(embedding_dim, embedding_dim)
        
        # Lệnh này thì là tạo nên một lớp gồm cả 3 thành phần ảnh , chữ hoặc đoạn văn và phần vector kết hợp của user và product 
        self.fusion = nn.Linear(embedding_dim * 3, embedding_dim)

    def forward(self, user_ids, product_ids, text_batch, edge_index, product_images_df=None):
        # Collaborative features
        """ Phần này là phần xử lý các thông tin liên quan đến các loại thông tin kết hợp về khách hàng và sản phẩm của cửa hàng """

        # Tạo vector embedding cho người dùng dựa trên bảng tra cứu 
        user_emb = self.user_emb(user_ids)

        # Tạo vector embedding dành cho sản phẩm dựa trên bảng tra cứu
        product_emb = self.product_emb(product_ids)
        
        #Lệnh này dùng để điều chỉnh cái bảng hiển thị mua sắm của khách hàng nếu như chỉ có một khách hàng mà mua nhiều loại
        #sản phẩm thì phải thêm một vài dòng trống ở chỗ user để cho cân đối
        if len(user_emb.shape) == 2 and len(product_emb.shape) == 2 and user_emb.shape[0] == 1:
            user_emb = user_emb.expand(product_emb.shape[0], -1)
        
        """ Phần này là xử lý thông tin về các loại hình ảnh """
        if product_images_df is not None:
            transform = transforms.Compose([# gộp các cái lệnh trong compose thì nó sẽ thực hiện đồng thời , tối ưu thời gian tốc độ
                # câu lệnh này dùng để chuyển size hình ảnh của hình ảnh ban đầu về size cố định đã được trained trong resnet50
                transforms.Resize((224, 224)),
                # chuyển hóa hình ảnh được truyền vào chuyển thành vector số học (vector embedding) để máy tình có thể đọc hiểu và xử lý 
                transforms.ToTensor(),
                # Reset50 là một model xử lí hình ảnh nên nó cần xự thống nhất về dữ liệu đầu vào, 
                #  Hình ảnh rất nhạy cảm với sự thay đổi về độ sáng, tương phản, màu sắc,
                # và thang giá trị pixel (0–255, 0–1, hay mean lệch nhau).
                # Thế nên lệnh này nên dùng để chuẩn hóa lại thang đo của vector embedding sẵn có để sao cho thống nhất lại 
                # tránh cho sự xung đột
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Tạo một dict với key là id sản phẩm tương ứng với các đường dẫn hình ảnh và góc nhìn tương ứng của ảnh đó 
            product_id_to_info = {}

            for _, row in product_images_df.iterrows():
                path = row['image_path']# tạo một biến lưu trữ lại đường link dẫn đến hình ảnh này , để sau lưu lại sau khi đổi lại view style của thời trang
                
                view_type = 0  # đặt sãn mặt định của style là góc nhìn từ trước tời (mặt trước)
                # các lệnh dưới thì nó sẽ chỉnh lại hình ảnh theo góc nhìn của hình ảnh mà chúng ta nạp vào
                if '_1_front' in path:
                    view_type = 0
                elif '_2_side' in path:
                    view_type = 1
                elif '_3_back' in path:
                    view_type = 2
                elif '_4_full' in path:
                    view_type = 3
                product_id_to_info[row['product_id']] = {'path': path, 'view_type': view_type}
                # với mỗi sản phẩm đã có thì nó sẽ có nhiều góc nhìn của sản phẩm, Anh_xa_hinh_anh dùng để luu lại các hình ảnh 
                # và cũng như phân loại, gắn nhãn dán của hình ảnh theo các góc nhìn 
            
           
            image_tensors = []
            view_types = []
            for pid in product_ids.cpu().numpy():
                # ở đây chúng ta phải chuyển sang numpy bởi vì Id_khach_hang hiện tại vẫn đang dưới 
                # dạng torch tensor , mà torch tensor thì nó lại ở trên GPU nên python bình thường không xử lý được dữ liệu trên GPU, nên chúng ta 
                # phải chuyển lại dữ liệu về trên cpu rồi sau đó mới chuyển lại cấu trúc dữ liệu về trên numpy 
                # mặc đù dữ liệu ở trên CPU thì vốn dĩ python đã có thể truy cập và sử dụng dữ liệu rồi nhưng vẫn phải chuyển về trên numpy 
                # bởi vì khi truy cập trực tiếp tren CPU thì dạng dữ liệu nó truy cập ra thì có một vài dữ liệu thì nó lại không tương thích với 
                # cấu trúc hàm mà mình định xây dựng 
                if pid in product_id_to_info:
                    info = product_id_to_info[pid]# ở đây thì ban đầu thì Id_khach_hang nó vốn dĩ ở dạng tensor nhưng sau khi được chuyển 
                    # từ GPU sang CPU rồi sang numpy thì mọi định dạng dữ liệu đề có dạng float hoặc int để có thể xử lý trên code 
                    img_path = info['path']
                    if os.path.exists(img_path):# ở đây os có tác dụng cho phép có thể giao tiếp với hệ diều hành , còn os.path là một lệnh con cho phép code có thể xử lý các đường link xử lý trên hệ điều hành 
                        try:
                            img = Image.open(img_path).convert('RGB')
                            # Image.open là lệnh cho phép truy cập trực tiếp vào trong ảnh ngay trong máy tính , và ở đây thì việc chuyển ảnh về dạng RGB 
                            # là một điều kiện bắt buộc bởi vì mặc dù ảnh ban đầu đều dạng img nhưng bảng màu nó sử dụng có dạng khác nhau thì việc đồng nhât 
                            # về bảng màu RGB để đồng nhất về dạng bảng màu có tròng resnet50 
                            img_tensor = transform(img)# sử dụng modek mà minh bulft bên trên đê chuyển hóa hình ảnh ban đầu thành dạng tensor 
                            image_tensors.append(img_tensor)# them tensor của ảnh vào list 
                            view_types.append(info['view_type'])# them goc nhin tương ứng của hình ảnh này
                            logger.debug(f"Successfully loaded image for product {pid}: {img_path}")
                            # lệnh này thì nó sẽ là giống như là một cái đánh dấu cho viêc các lệnh phía trước đã chạy hoàn thành 
                            # bởi vì phải chạy qua các lệnh trước try thì nó sẽ phải qua các lệnh trước thì nó mới đến lệnh này 
                            # lệnh này nó thông báo cho rằng các lệnh phía trước đã chạy rồi
                        except Exception as e:
                            # lệnh này thì chính là dung để thông báo rằng phần try thì lệnh này nó sẽ có lỗi ở chỗ nào 
                            # thông báo lỗi ở đâu để chúng ta sưa lại e thì là loại lỗi mà chúng ta lưu bên trên 
                            logger.error(f"Error loading image for product {pid}: {e}")
                            # bởi vì bị lỗi không load đc hình ảnh nên chúng ta phải tạo một khung hình ảnh sao cho khi đang chạy data 
                            # thì nó không bị lỗi và dừng bởi vì ảnh không load đc, chúng ta có thể bổ sung lại hình ảnh 
                            # thiếu hụt trước đó sau 
                            image_tensors.append(torch.zeros(3, 224, 224))
                            # nó không có ảnh thì cứ để góc nhìn đại là 0 đi
                            view_types.append(0)
                    else:
                         # khi mà không có đường dẫn ảnh thì nó cũng tạo một cái vector ảnh rỗng giống như phân trên 
                        logger.warning(f"Image path does not exist for product {pid}: {img_path}")
                        image_tensors.append(torch.zeros(3, 224, 224))
                        # phần này lý thuyết chỉ khác một phần đó là mức độ cảnh báo khi mà có sai lầm xảy ra 
                        # mức độ nghiêm trọng error thì nó sẽ có thể ảnh hưởng đến hoạt động của app
                        # mức độ nghiêm trọng cua warning thì nó sẽ chỉ cảnh báo sẽ bởi vì nó sẽ chỉ có một vài data thiếu khuyết nó sẽ không ảnh hưởng đến hoạt động của app
                        view_types.append(0)

                else:
                    # nếu không có ảnh trong phần produt_info thì cũng tạo nên một tensor ảnh rỗng 
                    logger.warning(f"No image mapping for product {pid}")
                    image_tensors.append(torch.zeros(3, 224, 224))
                    view_types.append(0)
            
            # lệnh stack thì nó chính là để gộp các tensor ảnh lại thành một lúc cho phép xử lý các ảnh này cùng một lúc thay vì chỉ chạy từng 
            # cái bên trong list và lệnh của .device thì nó là đưa lô(batch) về phần cứng nơi khai báo vector embedding của người dùng 
            # để tránh việc không tìm thấy dữ liệu và đê dễ dàng xử lý 
            image_batch = torch.stack(image_tensors).to(user_emb.device)
            # cái này thì chúng ta chuyển list góc nhìn ảnh thành tensor rồi sau đó chuyển tensor về phần cứng của vector người dùng 
            # để tiện làm việc
            view_batch = torch.tensor(view_types).to(user_emb.device)
            
            # chuyển hóa lô ảnh của mình từ dạng file tensor thành lô ảnh vector embedding của mình 
            base_image_emb = self.image_encoder(image_batch)
            # chuyển hóa lô góc nhìn từ dạng lô tensor thành lô góc nhìn embedding 
            view_emb = self.view_embedding(view_batch)
            
            # tạo nên một file phong cách gồm là kết hợp của lô ảnh embedding và lô góc nhìn của vector embedding 
            image_emb = self.style_projection(base_image_emb + view_emb)
        else:
            # tạo một tensor giả toàn số 0 với product_emp.shape[0] thì là số sản phẩm , product_emd.shape[1] thì là kích thước của vector embedding 
            # rồi chuyển vị trí dữ liệu lên phần cứng nơi mà chứa các dữ liệu của vector embedding của user
            # tạo một vector giả toàn số 0 thì cho rồi khi chạy qua lệnh bên trên nếu có thì thay thế vecor 0
            # nếu như không có vector thì nó vẫn tồn tại một vector 0 thì khi chạy qua nó tránh bị lỗi 
            image_emb = torch.zeros(product_emb.shape[0], product_emb.shape[1]).to(user_emb.device)
        
        # ở đây thì phải check xem lô của các văn bản thì nó có đang ở dạng list không , và check xem lô văn bản thì có rỗng không ,nếu cả 2 đều ổn thì sẽ chạy phần dưới
        if isinstance(text_batch, list) and len(text_batch) > 0:
            # tạo một vector embedding của của dạng text chuyển lô văn bản đang ở dạng list thành các vector embedding số học 
            # và convert_to_Tensor= True thì nó là chắc chắn rằng đầu ra của mình ở dưới dạng kết quả của pytorch
            # và rồi chuyển các vector embedding của mình thì nó sẽ chuyển về vị trị phần cứng của mình nơi chứa vector người dùng 
            text_features = self.text_encoder.encode(text_batch, convert_to_tensor=True).to(user_emb.device)
            # định dạng lại vector embedding thành dạng 128 chiều 
            text_emb = self.text_proj(text_features)
        else:
            #tạo một vector tensor thì cứ tạo một file toàn 0 thì để cho nếu như không có phần description thì code vẫn chạy qua 
            text_emb = torch.zeros(product_emb.shape[0], product_emb.shape[1]).to(user_emb.device)
        
        cf_emb = user_emb * product_emb  # tạo một vector embedidng dành cho thể hiện mối quan hệ của người dùng và sản phẩm , 
        # theo mức độ phù hợp của người dùng và sản phẩm 
        combined = torch.cat([cf_emb, image_emb, text_emb], dim=-1) # tạo một vector embedding bao gồm ,  bằng cách ghép ngang 
        # là kiểu ghép ngang thì nó sẽ là kiểu ghép thêm nhiều loại kiểu dữ liệu, gồm các loại như ảnh, text , ... 
        return self.fusion(combined) # kết quả là một vector embedding , thì cái này có nghĩa là khi mà các trộn các vector embedding 
        # như là collabrative, image , text thì khi mà nó trọn lại thì có nghĩa là nó sẽ đánh lại trọng số theo người dùng 
        # bởi vì mỗi người dùng thì nó có một ưu tiên riêng như là theo có người dựa vào ảnh nhiều hơn , có người thì dựa vào 
        # description , ...
