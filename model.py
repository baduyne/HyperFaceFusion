import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import numpy as np
import cv2
import pytorch_msssim

# SSIM Loss Function
ssim_fn = pytorch_msssim.SSIM()

def ssim_loss(pred, target):
    return 1 - ssim_fn(pred, target)

def gradient_loss(img):
    dx = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
    dy = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
    return torch.mean(dx) + torch.mean(dy)

def preprocess_image(img_path, size=(128, 128)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    return img

def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(113),
        transforms.ToTensor()
    ])

# ===== Dense Block =====
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv(x))
        return torch.cat([x, out], 1)

# ===== BSRDB Block =====
class BSRDB(nn.Module):
    def __init__(self, in_channels=16, growth_rate=16):
        super().__init__()
        self.db1 = DenseBlock(in_channels, growth_rate)
        self.db2 = DenseBlock(in_channels + growth_rate, growth_rate)
        self.db3 = DenseBlock(in_channels + 2 * growth_rate, growth_rate)
        self.conv = nn.Conv2d(in_channels + 3 * growth_rate, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.proj = nn.Conv2d(in_channels, 64, kernel_size=1)

    def forward(self, x):
        identity = x
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        out = self.relu(self.conv(out))
        identity_proj = self.proj(identity)
        return out + identity_proj

# ===== Encoder =====
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bsrdb = BSRDB(16, 16)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.bsrdb(x)

# ===== Feedback Block =====
class FeedbackBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=3, padding=1)

    def forward(self, prev_out):
        return self.conv(prev_out)

# ===== Decoder =====
class Decoder(nn.Module):
    def __init__(self, T=4):
        super().__init__()
        self.T = T
        self.conv_layers = nn.Sequential(
            nn.Conv2d(65, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1)
        )
        self.fb = FeedbackBlock()

    def forward(self, x):
        B, _, H, W = x.shape
        prev = torch.zeros((B, 1, H, W), device=x.device)
        for _ in range(self.T):
            inp = torch.cat([x, prev], dim=1)
            out = self.conv_layers(inp)
            prev = self.fb(out)
        return out

# ===== Transfer Layer =====
class TransferLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(1)
        self.resize = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.norm(x)
        return self.resize(x)

# ===== Classifier Head =====
class ClassifierHead(nn.Module):
    def __init__(self, embedding_dim=512, num_classes=113):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

# ===== Full HyperFace Pipeline =====
class HyperFacePipeline(nn.Module):
    def __init__(self, num_classes=113):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.transfer = TransferLayer()
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()

        for param in self.facenet.parameters():
            param.requires_grad = False

        self.classifier = ClassifierHead(num_classes=num_classes)

    def forward(self, ir, vis):
        a1, a2 = 0.8, 0.2
        ir_mix = a1 * ir + a2 * vis
        vis_mix = a2 * ir + a1 * vis

        f_ir = self.encoder(ir_mix)
        f_vis = self.encoder(vis_mix)
        fused = f_ir + f_vis

        fused_face = self.decoder(fused)
        transferred = self.transfer(fused_face)
        fused_rgb = transferred.repeat(1, 3, 1, 1)

        embeddings = self.facenet(fused_rgb)
        logits = self.classifier(embeddings)

        return logits, embeddings, fused_face
