import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import numpy as np
import cv2
import pytorch_msssim

ssim_fn = pytorch_msssim.SSIM(data_range=1.0, channel=1)  # Truyền data_range TẠI ĐÂY khi khởi tạo đối tượng SSIM
# -------------------------------

def pixel_loss(pred, target):
    return F.mse_loss(pred, target)

# Define gradient_loss (Facial Detail Preserving Loss - non-reference)
def gradient_loss(img):
    # Calculate gradients using central difference
    # For a 1-channel image [N, 1, H, W]
    dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    return torch.mean(dx) + torch.mean(dy)

# Define ssim_loss (make sure it passes data_range to ssim_fn if needed)
def ssim_loss(pred, target):
    # --- ĐÂY CŨNG LÀ DÒNG CẦN SỬA ĐỔI ---
    # Vì ssim_fn đã được khởi tạo với data_range=1.0 ở trên,
    # bạn không cần truyền lại data_range trong lời gọi này.
    return 1 - ssim_fn(pred, target) 
    # -----------------------------------

def preprocess_image(img_path, size=(128, 128)):
    """
    Tiền xử lý ảnh: đọc, resize, chuẩn hóa và chuyển sang tensor PyTorch.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]
    return img

# ===== Dense Block (Khối dày đặc) =====
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv(x))
        # Ghép nối đầu vào với đầu ra của Conv-ReLU
        return torch.cat([x, out], 1)

# ===== BSRDB Block (Bi-Scope Residual Dense Block) =====
class BSRDB(nn.Module):
    def __init__(self, in_channels=16, growth_rate=16):
        super().__init__()
        # Ba DenseBlock
        self.db1 = DenseBlock(in_channels, growth_rate)
        self.db2 = DenseBlock(in_channels + growth_rate, growth_rate)
        self.db3 = DenseBlock(in_channels + 2 * growth_rate, growth_rate)
        
        # Lớp tích chập cuối cùng để ánh xạ các đặc trưng dày đặc
        # Đầu vào là tổng số kênh sau 3 DenseBlock (in_channels + 3 * growth_rate)
        # Đầu ra là 64 kênh như mô tả của BSRDB trong bài báo
        self.conv = nn.Conv2d(in_channels + 3 * growth_rate, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Lớp chiếu (projection) 1x1 cho kết nối dư toàn cục (global residual)
        # Ánh xạ đầu vào ban đầu (identity) để khớp số kênh với đầu ra của conv
        self.proj = nn.Conv2d(in_channels, 64, kernel_size=1)

    def forward(self, x):
        identity = x # Lưu trữ đầu vào ban đầu cho kết nối dư toàn cục
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        out = self.relu(self.conv(out))
        
        # Thực hiện kết nối dư toàn cục
        identity_proj = self.proj(identity)
        return out + identity_proj

# ===== Encoder (Bộ mã hóa) =====
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # BSRDB sẽ nhận 16 kênh đầu vào và đầu ra 64 kênh
        self.bsrdb = BSRDB(16, 16) 

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.bsrdb(x) # Đầu ra của Encoder là 64 kênh

# ===== Feedback Block (Khối phản hồi) =====
class FeedbackBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Lớp tích chập này sẽ lấy đầu ra 1 kênh của bộ giải mã (fused_face)
        # và biến đổi nó thành 64 kênh để làm thông tin phản hồi cho lần lặp tiếp theo.
        self.conv = nn.Conv2d(1, 64, kernel_size=3, padding=1) 

    def forward(self, prev_out): # prev_out ở đây là đầu ra 1 kênh cuối cùng của Decoder
        return self.conv(prev_out)

# ===== Decoder (Bộ giải mã) =====
class Decoder(nn.Module):
    def __init__(self, T=4):
        super().__init__()
        self.T = T # Số lần lặp của quá trình phản hồi
        
        # Các lớp tích chập của bộ giải mã.
        # Đầu vào của lớp đầu tiên là sự ghép nối của đặc trưng từ Encoder (64 kênh)
        # và đặc trưng phản hồi 'prev' (cũng 64 kênh) => tổng 128 kênh.
        self.conv_layers = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True), # Đã điều chỉnh đầu vào thành 128 kênh
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1) # Lớp cuối cùng trả về 1 kênh cho ảnh tái tạo
        )
        self.fb = FeedbackBlock() # Khối phản hồi

    def forward(self, x): # x là đặc trưng hợp nhất từ Encoder (64 kênh)
        B, _, H, W = x.shape
        # Khởi tạo 'prev' (thông tin phản hồi) với cùng số kênh với x (64 kênh)
        # để có thể ghép nối trong các lần lặp tiếp theo.
        prev = torch.zeros((B, 64, H, W), device=x.device) 
        
        for i in range(self.T):
            # Ghép nối đặc trưng từ Encoder (x) và thông tin phản hồi (prev)
            # Tổng số kênh sẽ là 64 + 64 = 128.
            inp = torch.cat([x, prev], dim=1) 
            out = self.conv_layers(inp) # 'out' là đầu ra hiện tại của bộ giải mã (1 kênh)
            
            # Cập nhật 'prev' cho lần lặp tiếp theo thông qua FeedbackBlock.
            # Chỉ cập nhật nếu không phải là lần lặp cuối cùng.
            if i < self.T - 1: 
                prev = self.fb(out) 
            
        return out # Trả về ảnh khuôn mặt tái tạo cuối cùng (1 kênh)

# ===== Transfer Layer (Lớp chuyển đổi) =====
class TransferLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(1) # Chuẩn hóa theo lô cho ảnh 1 kênh
        # Thay đổi kích thước ảnh về 128x128 bằng nội suy song tuyến
        self.resize = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.norm(x)
        return self.resize(x)

# ===== Classifier Head (Đầu phân loại) =====
class ClassifierHead(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=113):
        super().__init__()
        # Chuỗi các lớp tuyến tính cho phân loại
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes) # Đầu ra là số lượng lớp (ID)
        )

    def forward(self, x):
        return self.classifier(x)

# ===== Full HyperFace Pipeline (Toàn bộ Pipeline HyperFace) =====
class HyperFacePipeline(nn.Module):
    def __init__(self, num_classes=113):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.transfer = TransferLayer()
        # Sử dụng FaceNet đã được huấn luyện trước trên VGGFace2
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()

        # Đóng băng các tham số của FaceNet, không cho phép huấn luyện lại
        for param in self.facenet.parameters():
            param.requires_grad = False

        self.classifier = ClassifierHead(num_classes=num_classes)

    def forward(self, ir, vis):
        # 1. Sơ đồ tiền hợp nhất (Pre-fusion scheme)
        a1, a2 = 0.8, 0.2
        ir_mix = a1 * ir + a2 * vis
        vis_mix = a2 * ir + a1 * vis

        # 2. Bộ mã hóa Siamese (Siamese Encoder)
        f_ir = self.encoder(ir_mix) # Đặc trưng từ ảnh IR đã trộn (64 kênh)
        f_vis = self.encoder(vis_mix) # Đặc trưng từ ảnh VIS đã trộn (64 kênh)
        
        # 3. Chiến lược hợp nhất đặc trưng (Summation-based Fusion)
        fused = f_ir + f_vis # Đặc trưng hợp nhất (64 kênh)

        # 4. Bộ giải mã kiểu phản hồi (Feedback-style Decoder) để tái tạo khuôn mặt
        fused_face = self.decoder(fused) # Ảnh khuôn mặt tái tạo (1 kênh)

        # 5. Lớp chuyển đổi (Transfer Layer)
        transferred = self.transfer(fused_face) # Ảnh 1 kênh, 128x128

        # 6. Chuẩn bị cho FaceNet (yêu cầu 3 kênh RGB)
        fused_rgb = transferred.repeat(1, 3, 1, 1) # Lặp lại kênh để tạo ảnh 3 kênh

        # 7. Giai đoạn nhận dạng (Recognition Stage) với FaceNet làm bộ trích xuất đặc trưng
        embeddings = self.facenet(fused_rgb) # Vector nhúng 512 chiều từ FaceNet

        # 8. Đầu phân loại (Classifier Head) để dự đoán ID
        logits = self.classifier(embeddings)

        return logits, embeddings, fused_face # Trả về logits (ID), embeddings và ảnh hợp nhất