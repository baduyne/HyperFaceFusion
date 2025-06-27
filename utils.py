# utils.py
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2

# SSIM Loss
import pytorch_msssim
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
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])