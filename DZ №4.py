import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from imageio import imread
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from glob import glob
from unet_model import UNet

writer = SummaryWriter(comment="Segmentation")

# Проверка количества файлов в директории изображений и масок
images = glob("./4th/stage1_train/*/images/*")
masks = glob("./4th/stage1_train/*/masks/*")
print(f"Number of images found: {len(images)}")
print(f"Number of masks found: {len(masks)}")

torch.manual_seed(2023)

# Загрузка путей к данным
paths = glob("*./deep-learning/4th/stage1_train")


class DSB2018(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = glob(self.paths[idx] + "/images/*")[0]
        mask_imgs = glob(self.paths[idx] + "/masks/*")
        img = imread(img_path)[:, :, 0:3]
        img = np.moveaxis(img, -1, 0)
        img = img / 255.0
        masks = [imread(f) / 255.0 for f in mask_imgs]
        final_mask = np.zeros(masks[0].shape)
        for m in masks:
            final_mask = np.logical_or(final_mask, m)
        final_mask = final_mask.astype(np.float32)
        img, final_mask = torch.tensor(img), torch.tensor(final_mask).unsqueeze(0)
        img = F.interpolate(img.unsqueeze(0), (256, 256))
        final_mask = F.interpolate(final_mask.unsqueeze(0), (256, 256))
        return img.type(torch.FloatTensor)[0], final_mask.type(torch.FloatTensor)[0]


dsb_data = DSB2018(paths)

# Проверка длины набора данных
if len(dsb_data) < 2:
    raise ValueError(f"Dataset is too small for splitting: {len(dsb_data)} samples available.")

train_split, test_split = torch.utils.data.random_split(dsb_data, [90, len(dsb_data) - 90])
train_seg_loader = DataLoader(train_split, batch_size=1, shuffle=True)
test_seg_loader = DataLoader(test_split, batch_size=1)

# Отладка размеров выборок
print(f"Total dataset size: {len(dsb_data)}")
print(f"Training set size: {len(train_split)}")
print(f"Test set size: {len(test_split)}")


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)

        self.decoder4 = self.conv_block(1024 + 512, 512)
        self.decoder3 = self.conv_block(512 + 256, 256)
        self.decoder2 = self.conv_block(256 + 128, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2))

        bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2))

        dec4 = self.decoder4(
            torch.cat((F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True), enc4), dim=1))
        dec3 = self.decoder3(
            torch.cat((F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True), enc3), dim=1))
        dec2 = self.decoder2(
            torch.cat((F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True), enc2), dim=1))
        dec1 = self.decoder1(
            torch.cat((F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True), enc1), dim=1))

        return self.final_conv(dec1)


# Инициализация модели U-Net
model = UNet(n_channels=3, n_classes=1)
loss_func = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)


model.apply(initialize_weights)

num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x, y in train_seg_loader:
        optimizer.zero_grad()
        prediction = model(x)
        loss = loss_func(prediction, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

# Сохранение обученной модели
torch.save(model.state_dict(), './unet.pt')
print("Model saved to unet.pt")
model.eval()
test_loss = 0
for x, y in test_seg_loader:
    with torch.no_grad():
        prediction = model(x)
        loss = loss_func(prediction, y)
        test_loss += loss.item()

print(f"Test Loss: {test_loss / len(test_seg_loader):.4f}")