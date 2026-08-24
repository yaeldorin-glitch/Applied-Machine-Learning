"""
Standalone local training script for Part II (image colorization) -- for
running on a machine with its own GPU, outside Colab (VS Code / a plain
terminal). Colab's session limits and CPU-only local testing are the
reason this exists: no artificial epoch/data cap here, meant to run for
as long as it needs to.

Everything under MODEL / LOSSES / DATASET below is copied verbatim from
notebooks/colorization_perceptual_loss.ipynb -- same architecture, same
color-weighted-L1 + VGG perceptual loss, same dominant-hue-balanced
sampling. Only the orchestration around it (argparse, GPU enforcement,
data acquisition for local disk instead of Colab, checkpoint directory)
is new. If you change the model/loss/dataset in the notebook later, port
the same change here too -- there is no shared import between them by
design, so the notebook keeps working with zero local setup (per the
project's CPU-only, no-Docker offline-friendly baseline).

Usage (see PARTNER_SETUP.md for full first-time setup):

    python train_local_gpu.py \
        --checkpoint-dir "G:\\My Drive\\colorization_checkpoints_v3" \
        --epochs 40 --coco-images 80000

First run (no checkpoint.pt yet in --checkpoint-dir): warm-starts model
WEIGHTS ONLY from --warm-start-weights (the real epoch-26 GPU run,
weights_epoch26_warm_start.pth, bundled alongside this script), with a
FRESH optimizer and a FRESH LR schedule, epoch counter reset to 0.
Weights-only (not the full optimizer/scheduler state) is deliberate:
this run trains on a much larger and differently-composed dataset than
the one that produced those weights (Natural Images + Flowers102 -> +
up to --coco-images COCO photos, added specifically for scene diversity
-- sky, water, grass, people -- that the flower-heavy pool was thin on,
especially blue and red), so the OLD optimizer momentum doesn't
describe this run's actual data. The learned weights still transfer
and are worth keeping; the optimizer/schedule state is not. This
mirrors the notebook's own warm-start pattern (loads a reference
model's weights only, trains its own fresh optimizer on top).

Training runs until --patience epochs pass with no val-loss improvement
(early stopping), up to a high --epochs ceiling that's not meant to be
reached in practice -- LR itself is reduced automatically on shorter
plateaus (--lr-patience) via ReduceLROnPlateau, so the run tapers off
and converges rather than stopping at an arbitrary epoch count with the
LR still high.

Every subsequent run (checkpoint.pt already exists) resumes normally:
full model + optimizer + scheduler state, exactly where it left off --
so an interrupted run (crash, reboot, closed terminal) only costs the
current incomplete epoch, same guarantee as the notebook's Drive-backed
resume.
"""

import argparse
import glob
import json
import os
import random
import time
import shutil
import subprocess
import sys
import sysconfig
import zipfile
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as T
from PIL import Image

IMG_SIZE = 256

# =====================================================================
# MODEL -- verbatim from the notebook (cell: "1) YOUR MODEL")
# =====================================================================

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


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))


class ColorizeModel(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.enc1 = ResBlock(1, base)
        self.enc2 = ResBlock(base, base * 2)
        self.enc3 = ResBlock(base * 2, base * 4)
        self.enc4 = ResBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResBlock(base * 8, base * 8)
        self.upconv4 = nn.ConvTranspose2d(base * 8, base * 8, kernel_size=2, stride=2)
        self.dec4 = ResBlock(base * 16, base * 4)
        self.upconv3 = nn.ConvTranspose2d(base * 4, base * 4, kernel_size=2, stride=2)
        self.dec3 = ResBlock(base * 8, base * 2)
        self.upconv2 = nn.ConvTranspose2d(base * 2, base * 2, kernel_size=2, stride=2)
        self.dec2 = ResBlock(base * 4, base)
        self.upconv1 = nn.ConvTranspose2d(base, base, kernel_size=2, stride=2)
        self.dec1 = ResBlock(base * 2, base)
        self.final_conv = nn.Conv2d(base, 2, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.upconv4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        return torch.sigmoid(self.final_conv(d1))


# =====================================================================
# LOSSES -- verbatim from the notebook (training cell)
# =====================================================================

def color_weighted_l1(pred_cbcr, target_cbcr, target_rgb):
    gray_target = target_rgb.mean(dim=1, keepdim=True)
    colorfulness = torch.abs(target_rgb - gray_target).mean(dim=1, keepdim=True)
    weight = 1.0 + 5.0 * colorfulness
    return (torch.abs(pred_cbcr - target_cbcr) * weight).mean()


class VGGPerceptualLoss(nn.Module):
    def __init__(self, layer_idx=26):
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


PERCEPTUAL_WEIGHT = 0.05

# =====================================================================
# DOMINANT-HUE CLASSIFICATION -- verbatim from the notebook (dataset cell)
# =====================================================================

HUE_BIN_EDGES = [(345, 15, "red"), (15, 45, "orange"), (45, 75, "yellow"),
                  (75, 165, "green"), (165, 195, "cyan"), (195, 255, "blue"),
                  (255, 285, "purple"), (285, 345, "magenta")]
NEUTRAL_THRESHOLD = 0.12


def get_hue_bins_cached(paths, cache_path, save_every=2000):
    """Same result as [dominant_hue_bin(p) for p in paths], but reads from
    a JSON cache (path -> bin) first and only classifies images not
    already in it -- makes every run after the first skip the ~102k-image
    classification scan entirely (was the last slow, uncached step before
    training could start; confirmed directly this was taking real time on
    a fresh run). Saves the cache to disk every save_every newly-classified
    images, not only once at the very end -- classifying ~102k images
    takes real minutes, and if this run gets interrupted partway through
    (closed terminal, crash, restart to change a setting -- all things
    that happened repeatedly today), only the images since the last
    periodic save need reclassifying, not the whole run from zero."""
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    bins = []
    new_count = 0
    total = len(paths)
    progress_every = max(1, total // 20)
    start = time.time()
    for i, p in enumerate(paths):
        if p in cache:
            bins.append(cache[p])
        else:
            b = dominant_hue_bin(p)
            cache[p] = b
            bins.append(b)
            new_count += 1
            if new_count % save_every == 0:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(cache, f)
        if (i + 1) % progress_every == 0 or (i + 1) == total:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_sec = (total - (i + 1)) / rate if rate > 0 else 0
            print("  hue classification: %d/%d (%.0f%%) -- %.0fs elapsed, ~%.0fs left"
                  % (i + 1, total, 100.0 * (i + 1) / total, elapsed, eta_sec))
    if new_count:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
        print("hue cache: classified %d new image(s), reused %d from cache" % (new_count, len(paths) - new_count))
    else:
        print("hue cache: all %d images found in cache, nothing to classify" % len(paths))
    return bins


def dominant_hue_bin(path):
    img = Image.open(path).convert("RGB").resize((64, 64))
    arr = np.asarray(img).astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    maxc, minc = np.max(arr, axis=-1), np.min(arr, axis=-1)
    v = maxc
    s = np.where(maxc > 0, (maxc - minc) / np.where(maxc == 0, 1, maxc), 0)
    delta = maxc - minc + 1e-8
    hue = np.zeros_like(maxc)
    mask_r = (maxc == r)
    mask_g = (maxc == g) & ~mask_r
    mask_b = (maxc == b) & ~mask_r & ~mask_g
    hue[mask_r] = (60 * (((g - b) / delta) % 6))[mask_r]
    hue[mask_g] = (60 * (((b - r) / delta) + 2))[mask_g]
    hue[mask_b] = (60 * (((r - g) / delta) + 4))[mask_b]
    weight = s * v
    if weight.mean() < NEUTRAL_THRESHOLD:
        return "neutral"
    best_bin, best_score = "neutral", 0.0
    for lo, hi, name in HUE_BIN_EDGES:
        band = (hue >= lo) | (hue <= hi) if lo > hi else (hue >= lo) & (hue <= hi)
        score = (weight * band).sum()
        if score > best_score:
            best_bin, best_score = name, score
    return best_bin


# =====================================================================
# DATASET -- verbatim from the notebook (dataset cell)
# =====================================================================

class ColorizationDataset(torch.utils.data.Dataset):
    def __init__(self, paths, size=IMG_SIZE, augment=False):
        self.paths = paths
        self.size = size
        self.resize = T.Resize((size, size))
        self.random_resized_crop = T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1))
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.augment:
            img = self.random_resized_crop(img)
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img = self.resize(img)
        rgb = np.array(img).astype("float32") / 255.0
        y, cb, cr = rgb_to_ycbcr(rgb)
        if self.augment:
            brightness = random.uniform(0.85, 1.15)
            contrast = random.uniform(0.85, 1.15)
            y = np.clip((y - 0.5) * contrast + 0.5, 0.0, 1.0)
            y = np.clip(y * brightness, 0.0, 1.0)
        y_t = torch.from_numpy(y).float().unsqueeze(0)
        cbcr_t = torch.from_numpy(np.stack([cb, cr], axis=0)).float()
        return y_t, cbcr_t


# =====================================================================
# DATA ACQUISITION -- new for local/VS Code use (Colab's auto-download
# flow doesn't apply here; same 3 sources, fetched to plain local disk)
# =====================================================================

def ensure_natural_images(data_root):
    natural_dir = os.path.join(data_root, "natural-images")
    if os.path.isdir(natural_dir) and glob.glob(os.path.join(natural_dir, "**", "*.jpg"), recursive=True):
        return sorted(glob.glob(os.path.join(natural_dir, "**", "*.jpg"), recursive=True))

    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        raise RuntimeError(
            "Natural Images dataset not found at %s, and no Kaggle API "
            "credentials at %s. Get a token from kaggle.com -> your "
            "profile -> Account -> Create New API Token, and place the "
            "downloaded kaggle.json at that path (see PARTNER_SETUP.md)."
            % (natural_dir, kaggle_json))
    # Invoke pip via sys.executable, and the kaggle CLI via its full install
    # path (sysconfig, not a bare "kaggle" on PATH) -- both a bare "pip" and
    # a bare "kaggle" command can fail to resolve on Windows even right
    # after installing successfully, since pip's own console-script
    # directory isn't always on PATH (confirmed directly: this exact
    # failure mode already hit "pip" itself during partner setup).
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)
    kaggle_exe = os.path.join(sysconfig.get_path("scripts"),
                               "kaggle.exe" if os.name == "nt" else "kaggle")
    subprocess.run([kaggle_exe, "datasets", "download", "-d", "prasunroy/natural-images",
                     "-p", data_root], check=True)
    zip_path = os.path.join(data_root, "natural-images.zip")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(natural_dir)
    print("Natural Images dataset downloaded and extracted.")
    paths = sorted(glob.glob(os.path.join(natural_dir, "**", "*.jpg"), recursive=True))
    if not paths:
        raise RuntimeError(
            "Natural Images extracted to %s but no .jpg files found -- "
            "the archive's folder layout may differ from what's expected. "
            "Check the actual contents of that directory." % natural_dir)
    return paths


def ensure_flowers102(data_root):
    flowers_dir = os.path.join(data_root, "flowers-102", "jpg")
    if not os.path.isdir(flowers_dir):
        for split in ["train", "val", "test"]:
            torchvision.datasets.Flowers102(root=data_root, split=split, download=True)
    return sorted(glob.glob(os.path.join(flowers_dir, "*.jpg")))


COCO_TRAIN2017_URL = "http://images.cocodataset.org/zips/train2017.zip"


def ensure_coco(data_root, max_images):
    """Up to `max_images` photos from COCO train2017 (118,287 total),
    added for scene diversity (sky, water, grass, streets, people) that
    Natural Images + Flowers102 alone are thin on -- directly the kind
    of content the blue-scarcity problem needs (measured: ~2.8% of the
    flower pool is dominant-blue, ~0.08% dominant-cyan; COCO photos are
    routinely 20-30% sky/water). max_images=0 skips this source
    entirely. The full zip (~19GB) is downloaded once regardless of
    max_images -- COCO ships as one archive, there's no partial-file
    download for a specific subset -- but only max_images members are
    actually extracted, chosen by a fixed-seed random sample so re-runs
    with the same max_images extract the same files instead of a new
    random subset each time.
    """
    if max_images <= 0:
        return []
    coco_dir = os.path.join(data_root, "coco-train2017")
    os.makedirs(coco_dir, exist_ok=True)
    existing = glob.glob(os.path.join(coco_dir, "*.jpg"))
    if len(existing) >= max_images:
        return sorted(existing)[:max_images]

    zip_path = os.path.join(data_root, "coco_train2017.zip")
    if not os.path.exists(zip_path):
        print("downloading COCO train2017 (~19GB, one-time) ...")
        subprocess.run(["curl", "-L", "-o", zip_path, COCO_TRAIN2017_URL], check=True)

    print("extracting up to %d COCO images ..." % max_images)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith(".jpg")]
        random.Random(0).shuffle(members)
        chosen = members[:max_images]
        for m in chosen:
            target = os.path.join(coco_dir, os.path.basename(m))
            if os.path.exists(target):
                continue
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    paths = sorted(glob.glob(os.path.join(coco_dir, "*.jpg")))
    print("COCO images ready:", len(paths))
    return paths


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="colorization_data_local",
                         help="where Natural Images / Flowers102 / COCO get downloaded to")
    parser.add_argument("--checkpoint-dir", required=True,
                         help="where weights.pth / weights_best.pth / checkpoint.pt are saved after every "
                              "epoch -- point this at your Google Drive Desktop synced folder "
                              "(e.g. \"G:\\My Drive\\colorization_checkpoints_v3\" on Windows) so every "
                              "save syncs automatically, same as the Colab run did")
    parser.add_argument("--warm-start-weights", default=os.path.join(os.path.dirname(__file__), "weights_epoch26_warm_start.pth"),
                         help="epoch-26 weights to warm-start from on a fresh run (ignored once checkpoint.pt exists)")
    parser.add_argument("--epochs", type=int, default=300,
                         help="hard ceiling on epochs -- in practice --patience (early stopping) is what "
                              "actually stops the run; this just bounds it in case val loss never plateaus")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--coco-images", type=int, default=80000,
                         help="how many COCO train2017 photos to add for scene/blue diversity; 0 to skip")
    parser.add_argument("--patience", type=int, default=6,
                         help="stop training if val loss hasn't improved for this many consecutive epochs "
                              "(0 disables early stopping and always runs the full --epochs). Must be "
                              "meaningfully larger than --lr-patience -- see the comment on the scheduler "
                              "below for why (otherwise the run stops in the same epoch the LR just got "
                              "cut, before that cut had any chance to help).")
    parser.add_argument("--lr-patience", type=int, default=2,
                         help="halve the LR if val loss hasn't improved for this many consecutive epochs "
                              "(should be smaller than --patience, so LR gets a chance to drop before giving up)")
    parser.add_argument("--allow-cpu", action="store_true",
                         help="without this flag, the script refuses to start unless CUDA is available -- "
                              "the whole point of running this locally instead of Colab is the GPU")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                         help="accumulate gradients over this many batches before each optimizer step. "
                              "Default of 1 (no accumulation) is correct when --batch-size is already 8 "
                              "(the default, and what the LR --lr 1e-3 was originally tuned for on Colab's "
                              "larger-VRAM GPUs). If you have to lower --batch-size for a smaller GPU (e.g. "
                              "4 on a 6GB card), raise this to compensate -- e.g. --batch-size 4 "
                              "--grad-accum-steps 2 makes each optimizer update based on 4*2=8 images' worth "
                              "of gradient again, while only ever holding 4 images in GPU memory at once. "
                              "Does NOT change BatchNorm's statistics, which are still computed per physical "
                              "batch -- this recovers batch=8-equivalent GRADIENT quality, not a fully "
                              "identical replica of true batch=8 in every respect.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise RuntimeError(
            "torch.cuda.is_available() is False -- this would train on CPU, which "
            "takes days-to-weeks longer than on your GPU and defeats the point of "
            "running this here instead of Colab. Most likely cause: pip installed "
            "the CPU-only torch build. See PARTNER_SETUP.md for the correct install "
            "command for your CUDA version, then re-run. If you really want a CPU "
            "run anyway (e.g. to test this script), pass --allow-cpu.")
    print("device:", device, ("(%s)" % torch.cuda.get_device_name(0)) if device.type == "cuda" else "")

    if device.type == "cuda":
        # Every image is resized to a fixed 256x256 -- input size never
        # varies, which is exactly the condition this flag needs to be a
        # safe, free speedup (cuDNN benchmarks conv algorithms once for
        # that fixed shape and reuses the fastest one, instead of picking
        # a generic one every call). Would be the wrong call with variable
        # input sizes; not a concern here.
        torch.backends.cudnn.benchmark = True

    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    natural_paths = ensure_natural_images(args.data_root)
    print("natural-images found:", len(natural_paths))
    flower_paths = ensure_flowers102(args.data_root)
    print("flowers102 found:", len(flower_paths))
    coco_paths = ensure_coco(args.data_root, args.coco_images)
    print("coco found:", len(coco_paths))

    all_paths = natural_paths + flower_paths + coco_paths
    print("classifying dominant hue for every image (one-time cost, cached after the first run) --",
          "%d images ..." % len(all_paths))
    hue_cache_path = os.path.join(args.data_root, "hue_cache.json")
    all_bins = get_hue_bins_cached(all_paths, hue_cache_path)
    bin_counts = Counter(all_bins)
    print("hue distribution across the full pool:", dict(bin_counts))

    combined = list(zip(all_paths, all_bins))
    random.Random(0).shuffle(combined)
    image_paths = [p for p, _ in combined]
    image_bins = [b for _, b in combined]

    n_val = max(1, int(0.1 * len(image_paths)))
    val_paths, train_paths = image_paths[:n_val], image_paths[n_val:]
    val_bins, train_bins = image_bins[:n_val], image_bins[n_val:]
    print("combined dataset:", len(image_paths), " train:", len(train_paths), " val:", len(val_paths))

    train_bin_counts = Counter(train_bins)
    train_weights = [1.0 / train_bin_counts[b] ** 0.5 for b in train_bins]
    print("train hue distribution:", dict(train_bin_counts))

    train_ds = ColorizationDataset(train_paths, augment=True)
    val_ds = ColorizationDataset(val_paths, augment=False)

    torch.manual_seed(0)
    model = ColorizeModel().to(device)

    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint.pt")
    weights_path = os.path.join(args.checkpoint_dir, "weights.pth")
    weights_best_path = os.path.join(args.checkpoint_dir, "weights_best.pth")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # ReduceLROnPlateau, not CosineAnnealingLR: the notebook's cosine schedule
    # is sized to a fixed EPOCHS because Colab sessions target a specific
    # session budget. Here --epochs is a high ceiling and --patience (early
    # stopping) is what actually ends the run, so a schedule tied to val-loss
    # plateaus fits better than one tied to a step count that may never be
    # reached -- otherwise the LR could still be sitting near its initial
    # value when early stopping fires, instead of having actually annealed
    # down for fine convergence.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=args.lr_patience)

    # Mixed precision: on RTX-class GPUs (Tensor Cores), running the
    # forward pass in float16 where safe is both faster and roughly
    # halves activation memory, while GradScaler keeps the backward pass
    # numerically stable (it scales the loss up before backward so small
    # gradients don't underflow to zero in float16, then unscales before
    # the optimizer step) -- this is what keeps AMP from silently
    # degrading training quality, not just a speed trick on its own.
    # enabled=False on CPU makes autocast/scaler a no-op automatically,
    # so --allow-cpu testing still works unchanged.
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_epoch = 0
    best_val_loss = float("inf")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print("resuming from checkpoint: epoch %d, best val loss so far %.4f" % (start_epoch, best_val_loss))
    elif os.path.exists(args.warm_start_weights):
        sd = torch.load(args.warm_start_weights, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        print("warm-started all weights from", args.warm_start_weights,
              "-- fresh optimizer/scheduler, epoch counter reset to 0")
    else:
        print("no checkpoint and no warm-start weights found at %s -- "
              "starting from random initialization" % args.warm_start_weights)

    vgg_loss_fn = VGGPerceptualLoss().to(device)

    train_sampler = torch.utils.data.WeightedRandomSampler(
        train_weights, num_samples=len(train_weights), replacement=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=(device.type == "cuda"), persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"), persistent_workers=True)

    epochs_since_improvement = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        opt.zero_grad()
        num_batches = 0
        num_train_batches = len(train_loader)
        # Print roughly 20 progress updates per epoch (every ~5%) -- with
        # tens of thousands of batches and nothing printed until the whole
        # epoch finishes, a run can look identical whether it's working
        # normally or silently stuck; this gives a live, visible signal
        # (batch count and elapsed time) during the wait instead of none.
        progress_every = max(1, num_train_batches // 20)
        epoch_start = time.time()
        for step, (y, cbcr) in enumerate(train_loader):
            y, cbcr = y.to(device), cbcr.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(y)
                pred_rgb = ycbcr_to_rgb_torch(y, pred[:, 0:1], pred[:, 1:2])
                target_rgb = ycbcr_to_rgb_torch(y, cbcr[:, 0:1], cbcr[:, 1:2])
                l1 = color_weighted_l1(pred, cbcr, target_rgb)
                perceptual = vgg_loss_fn(pred_rgb, target_rgb)
                loss = l1 + PERCEPTUAL_WEIGHT * perceptual
            # Divide by grad_accum_steps so accumulated gradients have the
            # same magnitude a single loss.backward() on the larger
            # effective batch would have produced (each micro-batch's
            # gradient contributes its proportional share, not a full share
            # grad_accum_steps times over). scaler.scale multiplies the
            # loss up before backward so small float16 gradients don't
            # underflow to zero; scaler.step unscales them back down
            # before actually calling opt.step (and skips the step
            # entirely if it finds an inf/nan, adjusting the scale for
            # next time instead of corrupting the weights).
            scaler.scale(loss / args.grad_accum_steps).backward()
            train_loss += loss.item() * y.size(0)
            num_batches = step + 1
            if num_batches % args.grad_accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            if num_batches % progress_every == 0 or num_batches == num_train_batches:
                elapsed = time.time() - epoch_start
                rate = num_batches / elapsed if elapsed > 0 else 0
                eta_sec = (num_train_batches - num_batches) / rate if rate > 0 else 0
                print("  epoch %d train: batch %d/%d (%.0f%%) -- %.0fs elapsed, ~%.0fs left in this epoch"
                      % (epoch, num_batches, num_train_batches, 100.0 * num_batches / num_train_batches,
                         elapsed, eta_sec))
        if num_batches % args.grad_accum_steps != 0:
            scaler.step(opt)  # flush leftover accumulation if the batch count doesn't divide evenly
            scaler.update()
            opt.zero_grad()
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            for y, cbcr in val_loader:
                y, cbcr = y.to(device), cbcr.to(device)
                pred = model(y)
                pred_rgb = ycbcr_to_rgb_torch(y, pred[:, 0:1], pred[:, 1:2])
                target_rgb = ycbcr_to_rgb_torch(y, cbcr[:, 0:1], cbcr[:, 1:2])
                l1 = color_weighted_l1(pred, cbcr, target_rgb)
                perceptual = vgg_loss_fn(pred_rgb, target_rgb)
                val_loss += (l1 + PERCEPTUAL_WEIGHT * perceptual).item() * y.size(0)
        val_loss /= len(val_ds)
        scheduler.step(val_loss)

        print("epoch %2d | lr %.2e | train loss %.4f | val loss %.4f" % (
            epoch, opt.param_groups[0]["lr"], train_loss, val_loss))

        torch.save(model.state_dict(), weights_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), weights_best_path)
            print("  (new best -- also saved to weights_best.pth)")
        else:
            epochs_since_improvement += 1

        torch.save({
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        }, checkpoint_path)

        if args.patience > 0 and epochs_since_improvement >= args.patience:
            print("early stopping: no val loss improvement for %d epochs (best %.4f, at epoch %d) -- "
                  "weights_best.pth already holds the best model, nothing more to gain by continuing."
                  % (args.patience, best_val_loss, epoch - args.patience))
            break

    print("training complete. best val loss: %.4f" % best_val_loss)
    print("weights.pth / weights_best.pth / checkpoint.pt are all in", args.checkpoint_dir)


if __name__ == "__main__":
    main()
