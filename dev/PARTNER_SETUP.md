# Setup guide -- running the colorization training on your GPU

This runs the same model that's been training in Google Colab, but on your
own machine so it can use your GPU directly (Colab's GPU quota is the
bottleneck we're trying to get around -- running the *notebook* in Colab
again would still use Colab's GPU, not yours, so that's not an option here).
It trains for as long as it takes to stop improving, saving progress after
every epoch to a folder that syncs to Google Drive automatically, so it can
be checked at any point without waiting for the whole run to finish.

You do not need to understand machine learning to do this -- just follow the
steps in order. Where a step matters ("don't skip this"), it says why.

---

## What you'll end up with

A terminal window running a training script for (realistically) anywhere
from several hours to a few days, depending on your GPU. It prints one line
per "epoch" (one full pass over the training photos) and saves 3 files to
your Google Drive after every single epoch:

- `weights.pth` -- the latest model, overwritten every epoch
- `weights_best.pth` -- the best model so far (only updates when it actually
  improves) -- **this is the one that matters**
- `checkpoint.pt` -- lets the script resume exactly where it left off if
  it's ever interrupted (closed terminal, restart, crash)

---

## Before you start: what you need from your partner (already being sent)

- A folder/zip containing: `train_local_gpu.py`, `requirements_gpu_partner.txt`,
  `weights_epoch26_warm_start.pth`, and this file.
- A Kaggle account credentials file (`kaggle.json`).
- A Google Drive folder shared with you, with a `weights.pth` file inside it
  -- move/copy that file into your own Drive's "My Drive" root area if you
  haven't already (you mentioned you already did this step).

---

## Step 1 -- Confirm you have an NVIDIA GPU

Open the Start menu, search "Device Manager", open it, expand "Display
adapters". If you see something like "NVIDIA GeForce/RTX/GTX ..." you're
good -- that's what all the steps below assume. (If it only shows Intel or
AMD, stop and let your partner know -- the CUDA-specific steps below won't
apply and we'd need a different plan.)

## Step 2 -- Make sure your NVIDIA drivers are current

Open a terminal (Start menu -> search "Terminal" or "PowerShell") and run:

```
nvidia-smi
```

If that prints a table with your GPU's name, driver version, and CUDA
version in the top right -- you're set, skip to Step 3. If it says the
command isn't recognized, search "NVIDIA driver download" on the NVIDIA
website, download the driver for your exact GPU model, install it, restart,
then re-run `nvidia-smi` to confirm.

## Step 3 -- Install Python (if VS Code doesn't already have one set up)

In VS Code: open the Extensions panel (left sidebar) and make sure the
"Python" extension (by Microsoft) is installed. Then open a terminal inside
VS Code (Terminal menu -> New Terminal) and run:

```
python --version
```

If it prints something like `Python 3.11.x` (anything 3.10-3.13 is fine),
you're set. If it errors, search "python.org downloads", get the latest
installer, run it (check the box that says "Add python.exe to PATH" during
install), then re-open the VS Code terminal and try again.

## Step 4 -- Install PyTorch WITH CUDA support (the step most likely to go wrong)

This is the one step to be careful with. Running `pip install torch` by
itself often installs a CPU-only build with no error message -- it just
silently trains on CPU instead of your GPU, which would turn an
overnight run into a multi-week one.

Go to **pytorch.org/get-started/locally** in a browser. It has a small
form: pick "Stable", your OS (Windows), "Pip", "Python", and for "Compute
Platform" pick the CUDA version that matches what `nvidia-smi` showed you in
Step 2 (pick the closest one at or below your driver's CUDA version). It
generates a command that looks like this (yours may differ -- use the one
the site gives you, not this exact line):

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Run that command in the VS Code terminal. This step downloads a few GB and
can take a while.

**Then verify it actually worked, before doing anything else:**

```
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

This must print `True` and your GPU's name. If it prints `False`, the
install grabbed the CPU build -- redo Step 4, double check the CUDA version
you picked, and don't move on until this prints `True`. The training script
itself also checks this automatically and refuses to start on CPU (it will
tell you if this step needs redoing), but it's worth confirming here first
so you're not waiting on the script to tell you.

## Step 5 -- Install the rest of the Python packages

In the folder your partner sent you, open a terminal in VS Code and run:

```
pip install -r requirements_gpu_partner.txt
```

(This deliberately does *not* touch torch/torchvision -- those are already
correctly installed from Step 4, and this file is designed not to
overwrite them.)

## Step 6 -- Install Google Drive for Desktop

Search "Google Drive for Desktop download" and install it (this is
different from just using drive.google.com in a browser -- it makes your
Drive appear as a normal folder on your computer, so the training script
can just save files into it like any other folder and have them sync
automatically). Sign in with the Google account your partner shared the
weights folder with.

Once it's running, open File Explorer -- you should see "Google Drive" in
the left sidebar. Click it and note the path (usually something like
`G:\My Drive\` -- it'll show in the File Explorer address bar). You'll need
this path in Step 9.

## Step 7 -- Place the Kaggle credentials file

Your partner is sending you a `kaggle.json` file. Put it here:

```
C:\Users\<your Windows username>\.kaggle\kaggle.json
```

(Create the `.kaggle` folder if it doesn't exist.) This lets the script
automatically download one of the training datasets (Kaggle "Natural
Images") -- without it, that download step will fail with a clear message
telling you this file is missing.

## Step 8 -- Locate the warm-start weights file

Confirm `weights_epoch26_warm_start.pth` (sent alongside the script) is in
the same folder as `train_local_gpu.py`. This is real progress from an
earlier training run -- the script starts from here instead of from
scratch, so hours of earlier training aren't wasted.

## Step 9 -- Run it

In the VS Code terminal, in the folder with `train_local_gpu.py`:

```
python train_local_gpu.py --checkpoint-dir "G:\My Drive\colorization_checkpoints_v3" --epochs 300 --coco-images 80000 --batch-size 4
```

Replace `G:\My Drive\` with whatever Step 6 showed you if it's different.
Note this uses a **new** folder name (`_v3`, not `_v2`) -- deliberately not
the same Drive folder the Colab run was using, so this run starts its own
clean training schedule instead of trying to resume the old run's optimizer
state, which was tuned for a much smaller dataset.

**`--batch-size 4`, not the script's default of 8**: the RTX 4050 laptop GPU
has 6GB of VRAM. The Colab runs this model was tuned on had far more (Colab
GPUs are typically 16GB+), and this model is memory-hungry -- a 15.3M-
parameter U-Net plus a VGG19 perceptual loss, both run at full 256x256. If
you still see a "CUDA out of memory" error at batch=4, add `--batch-size 2`
instead (see the troubleshooting section below).

## Step 10 -- What happens next, and how long it takes

In order, the first time you run it:
1. Downloads Flowers102 (~330MB) and the Kaggle Natural Images dataset
   (~350MB) -- a couple minutes.
2. Downloads COCO train2017 (~19GB) -- this is the big one, could be
   20 minutes to a few hours depending on your internet connection. This
   only happens once; it's cached locally after that.
3. Downloads pretrained VGG19 weights (~550MB, used for the perceptual
   loss) -- a minute or two.
4. Classifies every image's dominant color (a few minutes with ~95,000
   images total) -- prints one message before and after, no per-image
   output, that's normal, let it run.
5. Starts printing one line per epoch, e.g.:
   `epoch  5 | lr 1.00e-03 | train loss 0.1823 | val loss 0.1791`

**Rough estimate for an RTX 4050 laptop GPU, batch=4, ~95,000 images**:
somewhere in the range of 15-40 minutes per epoch -- this is a ballpark
from general knowledge of that GPU's class, not a measured number, so treat
it as a rough guide, not a promise. Watch how long the first 1-2 epochs
actually take once step 4 finishes and use that instead -- it's the real
number for this exact machine and dataset. Early stopping (see below) means
you don't need to estimate a total beforehand; it's normal for the whole
run to take anywhere from several hours to a couple of days unattended.

**It stops by itself (early stopping):** you don't need to watch it or
guess when to kill it. If val loss hasn't improved for 6 straight epochs,
it prints `early stopping: ...` and exits on its own -- `weights_best.pth`
on Drive already holds the best model at that point.

**Every epoch's save is what matters:** because `weights_best.pth` updates
on Drive after every improving epoch, your partner can download and check
progress at any time without waiting for the run to finish or interrupting
it.

## Step 11 -- Don't let it get interrupted

- Keep the VS Code terminal window open for the whole run. Closing it kills
  the process.
- In Windows Settings -> System -> Power, set "Sleep" to "Never" while
  plugged in (a sleeping laptop pauses everything, including this).
- If it does get interrupted for any reason, just run the exact same
  command again -- `checkpoint.pt` in the Drive folder lets it resume from
  the last completed epoch instead of starting over.

---

## If something goes wrong

- **"torch.cuda.is_available() is False"** when the script starts: redo
  Step 4, you likely have the CPU-only torch build installed.
- **"CUDA out of memory"** partway through an epoch: add `--batch-size 2` to
  the run command (or `--batch-size 1` if 2 still fails). This just means
  fewer images are processed at once -- it doesn't affect the final result,
  only makes each epoch a bit slower. Re-run the same command; `checkpoint.pt`
  (if any epoch already completed) or `weights_epoch26_warm_start.pth`
  (if none did) means nothing already done is lost.
- **Kaggle download fails / asks for credentials**: redo Step 7, check the
  file is at exactly `C:\Users\<you>\.kaggle\kaggle.json`.
- **Anything else**: copy the last ~30 lines of the terminal output and send
  them back -- that's almost always enough to tell what happened.
