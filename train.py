import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import rgb_to_grayscale # Đảm bảo import này

# Import các lớp mô hình và dataset của bạn
# Đảm bảo các hàm loss (pixel_loss, ssim_loss, gradient_loss)
# và class HyperFacePipeline được import từ model.py
from model import HyperFacePipeline, pixel_loss, ssim_loss, gradient_loss
from dataset import HyperspectralFaceDataset

# Load dataset
transform = transforms.ToTensor()
train_dataset = HyperspectralFaceDataset("./data/RGB_Thermal/rgb", "./data/RGB_Thermal/thermal", transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Khởi tạo Mô hình, Hàm mất mát và Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Số lớp thực tế của bạn là 113 (từ ID 1 đến 113), nên num_classes là 113
num_classes_actual = 113
model = HyperFacePipeline(num_classes=num_classes_actual).to(device)

# Hàm mất mát phân loại cho ID (CrossEntropyLoss)
criterion_cls = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Vòng lặp huấn luyện
num_epochs = 113
print(f"Bắt đầu huấn luyện trên {device}...")

for epoch in range(num_epochs):
    model.train() # Đặt mô hình ở chế độ huấn luyện
    total_composite_loss, correct, total = 0, 0, 0
    
    for batch_idx, (ir, vis, labels) in enumerate(train_loader):
        ir, vis, labels = ir.to(device), vis.to(device), labels.to(device)
        
        optimizer.zero_grad() # Xóa gradient của các tham số

        # --- ĐIỀU CHỈNH NHÃN VỀ 0-INDEXED ---
        labels = labels - 1 # Chuyển nhãn từ 1-113 thành 0-112
        # -----------------------------------
        
        # Lấy tất cả các đầu ra của mô hình: logits (cho phân loại), embeddings, và fused_face (ảnh tái tạo)
        logits, embeddings, fused_face = model(ir, vis)
        
        # --- Tính toán các thành phần hàm mất mát ---
        
        # 1. Hàm mất mát phân loại (L_CrossEntropy)
        loss_cls = criterion_cls(logits, labels)

        # 2. Hàm mất mát tái tạo (L_P, L_SS, L_FDP)
        # ĐẢM BẢO IR VÀ VIS ĐỀU LÀ 1 KÊNH TRƯỚC KHI TÍNH TRUNG BÌNH CỘNG
        # Kiểm tra số kênh hiện tại và chuyển đổi nếu cần
        if ir.shape[1] == 3: # Nếu ir là ảnh 3 kênh (RGB), chuyển về 1 kênh
            ir = rgb_to_grayscale(ir)
        
        if vis.shape[1] == 3: # Nếu vis là ảnh 3 kênh (RGB), chuyển về 1 kênh
            vis = rgb_to_grayscale(vis)
        
        # Tạo mục tiêu tái tạo bằng cách trung bình cộng của IR và VIS
        # Đảm bảo target_reconstruction cũng là 1 kênh
        target_reconstruction = (ir + vis) / 2 
        
        # Tính toán các thành phần loss tái tạo
        loss_pixel = pixel_loss(fused_face, target_reconstruction)
        
        # Gọi ssim_loss KHÔNG CÓ data_range ở đây, vì model.py đã xử lý việc này
        loss_ssim = ssim_loss(fused_face, target_reconstruction) 
        
        loss_gradient = gradient_loss(fused_face)

        # 3. Kết hợp các hàm mất mát thành hàm mất mát tổng hợp (Composite Loss)
        lambda_cls = 1.0  # Trọng số cho mất mát phân loại (có thể điều chỉnh)
        lambda_rec = 0.5  # Trọng số cho mất mát tái tạo (có thể điều chỉnh)
        
        composite_loss = (lambda_cls * loss_cls + 
                          lambda_rec * (loss_pixel + loss_ssim + loss_gradient))

        # Lan truyền ngược gradient và cập nhật tham số
        composite_loss.backward() # Sử dụng composite_loss để lan truyền ngược
        optimizer.step()

        # Cập nhật tổng mất mát và độ chính xác
        total_composite_loss += composite_loss.item() 
        
        _, preds = torch.max(logits, 1) # Sử dụng logits để lấy dự đoán lớp
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    # In kết quả sau mỗi epoch
    avg_composite_loss = total_composite_loss / len(train_loader)
    accuracy = correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Composite Loss: {avg_composite_loss:.4f}, Acc: {accuracy:.4f}")

print("Huấn luyện hoàn tất!")