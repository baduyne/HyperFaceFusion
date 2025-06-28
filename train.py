import torch
from torch.utils.data import DataLoader, random_split # Import random_split
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import rgb_to_grayscale
import numpy as np # Import numpy để xử lý ảnh và embeddings

# Import các lớp mô hình và dataset của bạn
from model import HyperFacePipeline, pixel_loss, ssim_loss, gradient_loss
from dataset import HyperspectralFaceDataset

# --- LỚP EARLY STOPPING ---
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='hyperface_best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss # Sử dụng âm của loss vì ta muốn loss càng nhỏ càng tốt

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# --- CÁC THAY ĐỔI TRONG PHẦN LOAD DATASET VÀ VÒNG LẶP HUẤN LUYỆN ---

# Load dataset
transform = transforms.ToTensor()
full_dataset = HyperspectralFaceDataset("./data/RGB_Thermal/rgb", "./data/RGB_Thermal/thermal", transform)

# Chia dataset thành tập huấn luyện và tập validation
train_size = int(0.8 * len(full_dataset)) # 80% cho huấn luyện
val_size = len(full_dataset) - train_size # 20% cho validation
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False) # Không shuffle tập validation

# Khởi tạo Mô hình, Hàm mất mát và Optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes_actual = 113
model = HyperFacePipeline(num_classes=num_classes_actual).to(device)

criterion_cls = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Khởi tạo Early Stopping
early_stopping = EarlyStopping(patience=10, verbose=True, path='hyperface_best_model.pth')

num_epochs = 113 # Số epoch tối đa

print(f"Bắt đầu huấn luyện trên {device}...")

for epoch in range(num_epochs):
    # --- Giai đoạn huấn luyện ---
    model.train()
    total_composite_loss, correct, total = 0, 0, 0
    
    for batch_idx, (ir, vis, labels) in enumerate(train_loader):
        ir, vis, labels = ir.to(device), vis.to(device), labels.to(device)
        optimizer.zero_grad()
        labels = labels - 1
        
        logits, embeddings, fused_face = model(ir, vis)
        
        loss_cls = criterion_cls(logits, labels)
        
        if ir.shape[1] == 3: 
            ir = rgb_to_grayscale(ir)
        if vis.shape[1] == 3: 
            vis = rgb_to_grayscale(vis)
        
        target_reconstruction = (ir + vis) / 2 
        loss_pixel = pixel_loss(fused_face, target_reconstruction)
        loss_ssim = ssim_loss(fused_face, target_reconstruction) 
        loss_gradient = gradient_loss(fused_face)

        lambda_cls = 1.0 
        lambda_rec = 0.5 
        composite_loss = (lambda_cls * loss_cls + 
                          lambda_rec * (loss_pixel + loss_ssim + loss_gradient))

        composite_loss.backward()
        optimizer.step()

        total_composite_loss += composite_loss.item() 
        _, preds = torch.max(logits, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    avg_train_composite_loss = total_composite_loss / len(train_loader)
    train_accuracy = correct / total
    
    # --- Giai đoạn đánh giá Validation ---
    model.eval() # Đặt mô hình ở chế độ đánh giá
    total_val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad(): # Không tính gradient trong quá trình validation
        for batch_idx_val, (ir_val, vis_val, labels_val) in enumerate(val_loader):
            ir_val, vis_val, labels_val = ir_val.to(device), vis_val.to(device), labels_val.to(device)
            labels_val = labels_val - 1 # Điều chỉnh nhãn

            logits_val, embeddings_val, fused_face_val = model(ir_val, vis_val)

            val_loss_cls = criterion_cls(logits_val, labels_val)

            if ir_val.shape[1] == 3: 
                ir_val = rgb_to_grayscale(ir_val)
            if vis_val.shape[1] == 3: 
                vis_val = rgb_to_grayscale(vis_val)
            
            target_reconstruction_val = (ir_val + vis_val) / 2
            val_loss_pixel = pixel_loss(fused_face_val, target_reconstruction_val)
            val_loss_ssim = ssim_loss(fused_face_val, target_reconstruction_val)
            val_loss_gradient = gradient_loss(fused_face_val)

            val_composite_loss = (lambda_cls * val_loss_cls + 
                                  lambda_rec * (val_loss_pixel + val_loss_ssim + val_loss_gradient))
            
            total_val_loss += val_composite_loss.item()
            _, val_preds = torch.max(logits_val, 1)
            val_correct += (val_preds == labels_val).sum().item()
            val_total += labels_val.size(0)
            
    avg_val_composite_loss = total_val_loss / len(val_loader)
    val_accuracy = val_correct / val_total

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_composite_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Val Loss: {avg_val_composite_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    # Gọi Early Stopping
    early_stopping(avg_val_composite_loss, model)
    if early_stopping.early_stop:
        print("Early stopping!")
        break # Dừng vòng lặp huấn luyện

print("Huấn luyện hoàn tất!")
print(f"Mô hình tốt nhất được lưu tại: {early_stopping.path}")