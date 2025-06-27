import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np 
class HyperspectralFaceDataset(Dataset):
    def __init__(self, rgb_dir, thermal_dir, transform=None):
        self.rgb_dir = rgb_dir
        self.thermal_dir = thermal_dir
        self.transform = transform

        # Lấy danh sách ảnh trong rgb_dir, sort để thứ tự cố định
        self.rgb_images = sorted(os.listdir(rgb_dir))

        # Lọc ra những ảnh có bản tương ứng trong thermal_dir (đảm bảo tồn tại)
        self.image_names = [img for img in self.rgb_images if os.path.exists(os.path.join(thermal_dir, img))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]

        # Load ảnh RGB và thermal
        rgb_path = os.path.join(self.rgb_dir, img_name)
        thermal_path = os.path.join(self.thermal_dir, img_name)

        rgb_img = Image.open(rgb_path).convert('L').resize((128, 128))
        thermal_img = Image.open(thermal_path).convert('L').resize((128, 128))

        # Lấy label là số đầu tiên trong tên file (trước dấu '-')
        label_str = img_name.split('-')[0]
        label = int(label_str)

        if self.transform:
            rgb_img = self.transform(rgb_img)
            thermal_img = self.transform(thermal_img)

        return rgb_img, thermal_img, label


# data_load = HyperspectralFaceDataset("./data/RGB_Thermal/rgb", "./data/RGB_Thermal/thermal")
# rgb_img, thermal_img, label = data_load.__getitem__(0)
# np_img = np.array(rgb_img)
# print(np_img.shape)


