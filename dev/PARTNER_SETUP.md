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

## Before you start: what you need

- The `dev` folder from the GitHub repo (or the standalone zip), containing:
  `train_local_gpu.py`, `PARTNER_SETUP.md` (this file), `requirements_gpu_partner.txt`,
  `weights_epoch26_warm_start.pth`. If you downloaded the whole repo as a
  ZIP from GitHub's "Code" button, these 4 files are inside its `dev`
  subfolder -- everything else in that download is unrelated (the other
  half of the project) and can be ignored.
- A Kaggle account credentials file (`kaggle.json`), sent separately.
- A Google Drive folder shared with you, with a `weights.pth` file inside it
  -- move/copy that file into your own Drive's "My Drive" root area if you
  haven't already.

You already confirmed you have an NVIDIA RTX 4050 -- that's a real GPU
PyTorch can use, so there's nothing to check there. Start with Step 1 below.

---

## Step 1 -- Check your NVIDIA driver, and note your CUDA version

Open a terminal (Start menu -> search "Terminal" or "PowerShell") and run:

```
nvidia-smi
```

If that prints a table with your GPU's name, driver version, and CUDA
version in the top right -- you're set, move to Step 2. Write down the CUDA
version shown (e.g. "12.4") -- you'll need it in Step 3. If it says the
command isn't recognized, search "NVIDIA driver download" on the NVIDIA
website, download the driver for your exact GPU model, install it, restart,
then re-run `nvidia-smi` to confirm.

## Step 2 -- Install Python 3.12 specifically (not the newest version)

**Use Python 3.12, not whatever's newest on python.org's front page.**
PyTorch's GPU-enabled Windows builds only support up to Python 3.12 --
anything newer (3.13, 3.14, ...) only has CPU-only wheels available, which
would silently defeat the entire point of running this on a GPU. Go to
**python.org/downloads/release/python-31210/**, scroll to "Files", and
download **"Windows installer (64-bit)"**. Run it, check the box that says
**"Add python.exe to PATH"**, click Install Now.

Then open a terminal inside VS Code (Terminal menu -> New Terminal, or
close and reopen it if one was already open) and run:

```
py --version
```

It should print `Python 3.12.10`. Use `py` (not `python`) for every command
below -- Windows ships a placeholder `python` command of its own that can
intercept the real one and silently do nothing useful (it just prints the
word "Python" with no version and no error); `py` reliably finds the actual
installed Python instead, so it's the safer command to use throughout.

## Step 3 -- Install PyTorch WITH the CUDA build (don't use the plain install)

On Windows, plain `pip install torch` reliably grabs a CPU-only build --
confirmed directly, not a maybe. Install the CUDA build explicitly instead:

```
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

(`cu126` was confirmed to have matching Windows/Python-3.12 wheels at the
time this guide was written. If this specific command ever fails with
"no matching distribution," go to **pytorch.org/get-started/locally** in a
browser -- it has a form that picks the current correct URL for you.)

This downloads a few GB, takes a few minutes. Then check it actually sees
your GPU:

```
py -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

**This must print `True` and your GPU's name before moving on.** If it
still prints `False`, run `py -m pip uninstall -y torch torchvision` first
(mixing a CPU build and a CUDA build in the same environment causes
confusing errors) and then retry the install command above. The training
script itself also refuses to start on CPU and will tell you if this step
needs redoing, but it's worth confirming here first rather than finding
out after everything else is set up.

## Step 4 -- Install the rest of the Python packages

In the folder your partner sent you, open a terminal in VS Code and run:

```
py -m pip install -r requirements_gpu_partner.txt
```

(This deliberately does *not* touch torch/torchvision -- those are already
correctly installed from Step 3, and this file is designed not to
overwrite them.)

## Step 5 -- Install Google Drive for Desktop

Search "Google Drive for Desktop download" and install it (this is
different from just using drive.google.com in a browser -- it makes your
Drive appear as a normal folder on your computer, so the training script
can just save files into it like any other folder and have them sync
automatically). Sign in with the Google account your partner shared the
weights folder with.

Once it's running, open File Explorer -- you should see "Google Drive" in
the left sidebar. Click it and note the path (usually something like
`G:\My Drive\` -- it'll show in the File Explorer address bar). You'll need
this path in Step 8.

## Step 6 -- Place the Kaggle credentials file

Your partner is sending you a `kaggle.json` file. Put it here:

```
C:\Users\<your Windows username>\.kaggle\kaggle.json
```

(Create the `.kaggle` folder if it doesn't exist.) This lets the script
automatically download one of the training datasets (Kaggle "Natural
Images") -- without it, that download step will fail with a clear message
telling you this file is missing.

## Step 7 -- Locate the warm-start weights file

Confirm `weights_epoch26_warm_start.pth` (sent alongside the script) is in
the same folder as `train_local_gpu.py`. This is real progress from an
earlier training run -- the script starts from here instead of from
scratch, so hours of earlier training aren't wasted.

## Step 8 -- Run it

In the VS Code terminal, in the folder with `train_local_gpu.py`:

```
py train_local_gpu.py --checkpoint-dir "G:\My Drive\colorization_checkpoints_v3" --epochs 300 --coco-images 80000 --batch-size 8 --grad-accum-steps 1
```

Replace `G:\My Drive\` with whatever Step 5 showed you if it's different.
Note this uses a **new** folder name (`_v3`, not `_v2`) -- deliberately not
the same Drive folder the Colab run was using, so this run starts its own
clean training schedule instead of trying to resume the old run's optimizer
state, which was tuned for a much smaller dataset.

**`--batch-size 8` -- confirmed to fit on a 6GB card, verified directly**:
memory usage at batch=4 measured at 2,921 MiB out of 6,141 MiB (under half),
so batch=8 has real headroom. If you still see a "CUDA out of memory" error,
drop to `--batch-size 4 --grad-accum-steps 2` instead (matches the same
effective gradient quality via accumulation, just processes fewer images in
memory at once -- see the troubleshooting section below), and only go as
low as `--batch-size 2 --grad-accum-steps 4` if that still isn't enough.

## Step 9 -- What happens next, and how long it takes

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

## Step 10 -- Don't let it get interrupted

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
  Step 3, you likely have the CPU-only torch build installed.
- **"CUDA out of memory"** partway through an epoch: lower `--batch-size`
  and raise `--grad-accum-steps` to compensate, keeping their product at 8
  -- try `--batch-size 4 --grad-accum-steps 2` first, then
  `--batch-size 2 --grad-accum-steps 4` if that's still too much. This
  keeps the same effective gradient quality while only holding fewer
  images in GPU memory at once -- it doesn't affect the final result, only
  makes each epoch a bit slower. Re-run the same command; `checkpoint.pt`
  (if any epoch already completed) or `weights_epoch26_warm_start.pth`
  (if none did) means nothing already done is lost.
- **Kaggle download fails / asks for credentials**: redo Step 6, check the
  file is at exactly `C:\Users\<you>\.kaggle\kaggle.json`.
- **Anything else**: copy the last ~30 lines of the terminal output and send
  them back -- that's almost always enough to tell what happened.
