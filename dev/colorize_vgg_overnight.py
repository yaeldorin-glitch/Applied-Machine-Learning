import os, glob, random, time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image

IMG_SIZE = 256
Y_R, Y_G, Y_B = 0.299, 0.587, 0.114


def rgb_to_ycbcr(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = Y_R * r + Y_G * g + Y_B * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 0.5
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 0.5
    return y, cb, cr


def ycbcr_to_rgb_torch(y, cb, cr):
    cb0, cr0 = cb - 0.5, cr - 0.5
    r = y + 1.402 * cr0
    g = y - 0.344136 * cb0 - 0.714136 * cr0
    b = y + 1.772 * cb0
    return torch.clamp(torch.cat([r, g, b], dim=1), 0.0, 1.0)


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


class ColorizeModel(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.enc1 = conv_block(1, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.enc4 = conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = conv_block(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.out = nn.Conv2d(base, 2, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))


class VGGPerceptualLoss(nn.Module):
    # Shallow slice (relu2_2) -- deep enough for real texture/color signal,
    # cheap enough to be feasible on CPU for an overnight run (benchmarked:
    # ~6.1s/batch at 6 threads vs ~9s+ for a mid-depth slice).
    def __init__(self, layer_idx=9):
        super().__init__()
        weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
        vgg = torchvision.models.vgg19(weights=weights).features[:layer_idx].eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred_rgb, target_rgb):
        pred_n = (pred_rgb - self.mean) / self.std
        target_n = (target_rgb - self.mean) / self.std
        return nn.functional.l1_loss(self.vgg(pred_n), self.vgg(target_n))


DATA_ROOT = "colorization_data"
IMAGE_DIR = os.path.join(DATA_ROOT, "flowers-102", "jpg")
image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
print("images found:", len(image_paths), flush=True)

random.Random(0).shuffle(image_paths)
N_IMAGES = 4000
image_paths = image_paths[:N_IMAGES]
n_val = max(1, int(0.1 * len(image_paths)))
val_paths, train_paths = image_paths[:n_val], image_paths[n_val:]
print("train:", len(train_paths), " val:", len(val_paths), flush=True)


class ColorizationDataset(torch.utils.data.Dataset):
    def __init__(self, paths, size=IMG_SIZE, augment=False):
        self.paths = paths
        self.resize = T.Resize((size, size))
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = self.resize(Image.open(self.paths[idx]).convert("RGB"))
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        rgb = np.array(img).astype("float32") / 255.0
        y, cb, cr = rgb_to_ycbcr(rgb)
        y_t = torch.from_numpy(y).float().unsqueeze(0)
        cbcr_t = torch.from_numpy(np.stack([cb, cr], axis=0)).float()
        return y_t, cbcr_t


train_ds = ColorizationDataset(train_paths, augment=True)
val_ds = ColorizationDataset(val_paths, augment=False)

BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-3
PERCEPTUAL_WEIGHT = 0.05

torch.set_num_threads(6)
torch.manual_seed(0)
model = ColorizeModel(base=24)
opt = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
l1_loss_fn = nn.L1Loss()
vgg_loss_fn = VGGPerceptualLoss()

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
n_batches = len(train_loader)
print("starting training: %d epochs, %d batches/epoch, PERCEPTUAL_WEIGHT=%.2f" % (EPOCHS, n_batches, PERCEPTUAL_WEIGHT), flush=True)

t_start = time.time()
best_val_loss = float("inf")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    t_epoch = time.time()
    for step, (y, cbcr) in enumerate(train_loader):
        opt.zero_grad()
        pred = model(y)
        l1 = l1_loss_fn(pred, cbcr)
        pred_rgb = ycbcr_to_rgb_torch(y, pred[:, 0:1], pred[:, 1:2])
        target_rgb = ycbcr_to_rgb_torch(y, cbcr[:, 0:1], cbcr[:, 1:2])
        perceptual = vgg_loss_fn(pred_rgb, target_rgb)
        loss = l1 + PERCEPTUAL_WEIGHT * perceptual
        loss.backward()
        opt.step()
        train_loss += loss.item() * y.size(0)
        if step % 20 == 0 or step == n_batches - 1:
            elapsed = time.time() - t_start
            print("  epoch %2d step %3d/%d | l1 %.4f perceptual %.4f | elapsed %.1fmin" % (
                epoch, step, n_batches, l1.item(), perceptual.item(), elapsed / 60), flush=True)
    train_loss /= len(train_ds)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for y, cbcr in val_loader:
            pred = model(y)
            l1 = l1_loss_fn(pred, cbcr)
            pred_rgb = ycbcr_to_rgb_torch(y, pred[:, 0:1], pred[:, 1:2])
            target_rgb = ycbcr_to_rgb_torch(y, cbcr[:, 0:1], cbcr[:, 1:2])
            perceptual = vgg_loss_fn(pred_rgb, target_rgb)
            val_loss += (l1 + PERCEPTUAL_WEIGHT * perceptual).item() * y.size(0)
    val_loss /= len(val_ds)
    scheduler.step()

    print("epoch %2d DONE | train loss %.4f | val loss %.4f | epoch time %.1fmin | total %.1fmin" % (
        epoch, train_loss, val_loss, (time.time() - t_epoch) / 60, (time.time() - t_start) / 60), flush=True)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "weights_vgg_backup.pth")
        print("  (checkpoint saved to weights_vgg_backup.pth)", flush=True)

print("TRAINING COMPLETE. total time %.1fmin, best val loss %.4f" % ((time.time() - t_start) / 60, best_val_loss), flush=True)
