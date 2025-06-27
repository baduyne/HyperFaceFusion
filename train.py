import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from model import *
from dataset import *
# Load dataset
transform = transforms.ToTensor()
train_dataset = HyperspectralFaceDataset("./data/RGB_Thermal/rgb", "./data/RGB_Thermal/thermal", transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HyperFacePipeline(113).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train loop
for epoch in range(113):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for ir, vis, labels in train_loader:
        ir, vis, labels = ir.to(device), vis.to(device), labels.to(device)
        optimizer.zero_grad()
        _, outputs = model(ir, vis)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Acc: {correct/total:.4f}")
