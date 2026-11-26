import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from scipy import ndimage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Geometry
DETECTOR_WIDTH_CM = 40.0    # width of the 64x64 detector mesh, cm
MAGNIFICATION = 1.25 # the source is a point at z = 0, so a crack's shadow spreads out on
# its way to the detector. Crack near the shield's back face (z ~ 30.5) and detector
# at z ~ 38 gives 38 / 30.5 ~ 1.25
PIXEL_SIZE_CM = DETECTOR_WIDTH_CM / 64 / MAGNIFICATION
ORIGIN_PX = (31.5, 31.5)    # (col, row) pixel position of x = 0, y = 0
# Orientation (found by geometry_scan.py): +x points right (col increases) and
# +y points DOWN the image (row increases), so cm and pixel axes point the same
# way and angles need no flipping.
MIN_WIDTH_PX = 2.0          # cracks thinner than this are drawn this wide

num_epochs = 150
batch_size = 16
learning_rate = 0.001
noise_std = 0.2
POS_WEIGHT = 50.0           # BCE weight on crack pixels
# Paths are anchored to the repo root (two levels above this file), so the script
# works no matter which folder it is launched from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.environ.get('DATA_DIR', os.path.join(REPO_ROOT, 'data'))
out_csv = os.environ.get('OUT_CSV', os.path.join(REPO_ROOT, 'crack_params_unet.csv'))
PARAM_NAMES = ['x', 'y', 'width', 'length', 'depth', 'phi']   # columns of the TRUE params
PRED_NAMES = ['x', 'y', 'length', 'phi'] # model unable to predict width or depth yet
PRED_COLS = [PARAM_NAMES.index(n) for n in PRED_NAMES]       # [0, 1, 3, 5]: where each sits in params

TRAIN = False                # True: train and save the models. False: load the saved models and just measure.
                             # WARNING: True overwrites the files at MODEL_PATH (your current best ensemble).

THRESHOLD = 0.5 #strict mask -> centre (x, y), angle (phi), width.
LENGTH_THRESHOLD = 0.25 # looser mask -> length only.

EVAL_ON = 'test'             # 'val' while comparing settings; 'test' once, for the final score

# Detection ("is there a crack at all?") is a separate decision from THRESHOLD, which
# only decides which pixels belong to the crack when measuring it. A wall is called
# cracked if crack_score > a cut-off, and the cut-off is chosen on the CLEAN VAL walls
# so that only a set fraction of them would be flagged (the false-alarm rate).
# The same cut-off is then applied unchanged to EVAL_ON, so test stays a fair score.
FA_TARGETS = (0.01, 0.05, 0.10)   # false-alarm rates to report
DETECT_FA = 0.05                  # the one used for the 'detected' column in the CSV

# Ensemble: train N_MODELS U-Nets that differ only in their random seed (starting weights, batch order, flips, noise)
# average their crack-probability maps.
N_MODELS = 5
MODEL_PATH = os.path.join(REPO_ROOT, 'unet_unc_seed{}.pt')   # one file per seed

def load_runs(data_dir):
    """
    Load every run_XXXXX.npz in data_dir.
    Returns
        run_ids   : list of str, e.g. 'run_00074'
        images    : (N, 3, H, W) float32 tensor: peak, scattered, uncollided
        has_crack : (N,) bool tensor
        params    : (N, 6) float tensor, columns PARAM_NAMES, NaN for clean walls.
                    phi in degrees [0, 180) = atan2(sin2phi, cos2phi) / 2.
    """
    images, has_crack, params, run_ids = [], [], [], []
    for i in range(0, 601):                       # run_00000 .. run_00600
        run_i = os.path.join(data_dir, f'run_{i:05d}.npz')
        if not os.path.exists(run_i):
            print(f'skipping missing {run_i}')
            continue
        with np.load(run_i) as f:
            image = torch.tensor(f['image'], dtype=torch.float32)               # (2, H, W): peak, scattered
            unc = torch.tensor(f['uncollided'], dtype=torch.float32)[None]      # (1, H, W)
            images.append(torch.cat([image, unc], dim=0))                       # (3, H, W)
            has_crack.append(bool(f['has_crack']))
            params.append(torch.tensor(f['params'], dtype=torch.float32))   # (7,), all NaN if clean
            run_ids.append(f'run_{i:05d}')

    images = torch.stack(images) #(N, 3, H, W): peak, scattered, uncollided
    has_crack = torch.tensor(has_crack) #(N,)
    params = torch.stack(params) # (N, 7,)

    phi = torch.rad2deg(torch.atan2(params[:, 5], params[:, 6])) / 2 % 180
    params = torch.cat([params[:, :5], phi[:, None]], dim=1)
    return images, has_crack, run_ids, params

def split_indices(has_crack):
    """
    70/15/15 stratified split
    Returns idx_train, idx_val, idx_test (1-D tensors of indices).
    """
    idx = np.arange(len(has_crack))
    idx_rest, idx_test = train_test_split(
        idx, test_size=0.15, random_state=1234, stratify=has_crack.numpy())
    idx_train, idx_val = train_test_split(
        idx_rest, test_size=0.15 / 0.85, random_state=1234,
    stratify=has_crack.numpy()[idx_rest])
    idx_train, idx_val, idx_test = (torch.as_tensor(a) for a in (idx_train, idx_val, idx_test))
    return idx_train, idx_val, idx_test


def preprocess(images, has_crack, idx_train):
    """
    Same input preprocessing as the classifier: log(flux / clean-wall reference),
    clamp to [-1, 1], per-channel standardise using training walls only.

    Every step works per channel, so it handles any number of channels unchanged.

    Returns X: (N, C, H, W) float tensor (C = 3: peak, scattered, uncollided).
    """
    clean_train = idx_train[~has_crack[idx_train]]
    assert len(clean_train) > 0, 'need at least one clean wall in the training set'
    ref = images[clean_train].mean(dim=0)  # (C, H, W), clean-wall average per channel
    ratio = (images / ref).clamp_min(1e-3)                      # guard against empty pixels
    X = torch.log(ratio)                                        # ~0 on clean walls, >0 at cracks
    n_clamped = int((ratio <= 1e-3).sum())
    # Real crack signal in 5 cm of concrete is at most ~2.4x (log 0.88); cap anything
    # beyond +-1 so a few near-empty pixels can't become huge outliers
    X = X.clamp(-1.0, 1.0)
    ch_mean = X[idx_train].mean(dim=(0, 2, 3), keepdim=True)    # (1, C, 1, 1): one per channel
    ch_std = X[idx_train].std(dim=(0, 2, 3), keepdim=True)
    X = (X - ch_mean) / (ch_std + 1e-8)
    y = has_crack.float()
    return X, y


def params_to_pixels(x, y, length, width, phi_deg):
    """
    Convert one crack from cm to pixel units.
    Returns (col, row, length_px, width_px, phi_px_rad):
        col, row    : crack centre in pixel coordinates (col = across, row = down)
        length_px   : length in pixels
        width_px    : max(width in pixels, MIN_WIDTH_PX)
        phi_px_rad  : angle of the crack's long axis in IMAGE coordinates, radians.
    """
    col = ORIGIN_PX[0] + x / PIXEL_SIZE_CM
    row = ORIGIN_PX[1] + y / PIXEL_SIZE_CM
    length_px = length / PIXEL_SIZE_CM
    width_px = max(width / PIXEL_SIZE_CM, MIN_WIDTH_PX)
    phi_px_rad = np.deg2rad(phi_deg)
    return col, row, length_px, width_px, phi_px_rad


def pixels_to_params(col, row, length_px, width_px, phi_px_rad):
    """
    inverse of params_to_pixels
    Returns (x, y, length, width, phi_deg) in cm / degrees, phi_deg in [0, 180]).
    """
    x = (col - ORIGIN_PX[0]) * PIXEL_SIZE_CM
    y = (row - ORIGIN_PX[1]) * PIXEL_SIZE_CM
    length = length_px * PIXEL_SIZE_CM
    width = width_px * PIXEL_SIZE_CM
    phi_deg = np.rad2deg(phi_px_rad) % 180
    return x, y, length, width, phi_deg


def draw_crack_mask(col, row, length_px, width_px, phi_px_rad, H=64, W=64):
    """
    Draw one crack (a rotated rectangle) as an (H, W) float tensor of 0s and 1s.
    """
    rows, cols = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                                torch.arange(W, dtype=torch.float32), indexing='ij')
    d_col = cols - col
    d_row = rows - row
    l_c = d_col * np.cos(phi_px_rad) + d_row * np.sin(phi_px_rad)
    w_c = -d_col * np.sin(phi_px_rad) + d_row * np.cos(phi_px_rad)
    inside = (l_c.abs() <= length_px / 2) & (w_c.abs() <= width_px / 2) # (H, W) True/False
    return inside.float() # (H, W) 1.0 / 0.0




def build_masks(params, has_crack, H, W):
    """
    One mask per run: (N, 1, H, W) float tensor. Clean walls are all zeros.
    """
    mask = torch.zeros(len(has_crack), 1, H, W)
    for i in range(len(has_crack)):
        if not has_crack[i]:
            continue                                  # clean wall: leave mask empty
        x, y, width, length, depth, phi = params[i].tolist()
        pixel_params = params_to_pixels(x, y, length=length, width=width, phi_deg=phi)
        mask[i,0] = draw_crack_mask(*pixel_params, H=H, W=W) # (N,1,H,W)
    return mask


def mask_to_params(prob, threshold=0.5):
    """
    Measure a crack from ONE predicted probability mask prob: (H, W) tensor in [0, 1].

    Returns (col, row, length_px, width_px, phi_px_rad), or None if no pixel > threshold.

    Hint (image moments): take the pixels above threshold (weighted by prob),
      centre      = weighted mean of their (col, row)
      orientation = direction of the largest eigenvector of their 2x2 covariance
      length/width: project pixels onto that direction (and the perpendicular one);
                    for a uniform rectangle, length = sqrt(12 * variance along it).
    """
    binary = (prob > threshold).numpy()
    labels, n = ndimage.label(binary)              # each separate blob gets a number 1..n
    if n == 0:
        return None
    sizes = ndimage.sum(binary, labels, range(1, n + 1))   # pixel count of each blob
    keep = labels == (np.argmax(sizes) + 1)                # True only on the biggest blob
    rows, cols = torch.where(torch.from_numpy(keep))
    #rows, cols = torch.where(prob > threshold)    # integer indices of crack pixels
    if len(rows) == 0:
        return None
    weights = prob[rows, cols]
    rows, cols = rows.float(), cols.float()
    col = (weights * cols).sum() / weights.sum()  # weighted mean column
    row = (weights * rows).sum() / weights.sum()  # weighted mean row
    d_col = cols - col
    d_row = rows - row
    W = weights.sum()
    c_cc = (weights * d_col * d_col).sum() / W
    c_rr = (weights * d_row * d_row).sum() / W
    c_cr = (weights * d_col * d_row).sum() / W
    cov = torch.stack([torch.stack([c_cc, c_cr]),   # wighted cov matrix
                       torch.stack([c_cr, c_rr])])
    evals, evecs = torch.linalg.eigh(cov)   # eigenvalues sorted ascending
    e = evecs[:, 1]                          # eigenvector for the largest one = along the crack
    phi = torch.atan2(e[1], e[0])
    u = d_col * torch.cos(phi) + d_row * torch.sin(phi)
    v = -d_col * torch.sin(phi) + d_row * torch.cos(phi)
    length_px = torch.sqrt(12 * (weights * u**2).sum() / W + 1)
    width_px  = torch.sqrt(12 * (weights * v**2).sum() / W + 1)
    return col.item(), row.item(), length_px.item(), width_px.item(), phi.item()

# 4. The U-Net
class DoubleConv(nn.Module):
    """
    (Conv 3x3 -> BatchNorm -> ReLU) twice. padding=1 so H and W don't change.
    (B, in_ch, H, W) -> (B, out_ch, H, W)
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    3-level U-Net for 64x64 inputs, `base` channels at the top level.
    forward(x) returns mask_logits (B, 1, H, W).
    """
    def __init__(self, in_ch=3, out_ch=1, features=(8, 16, 32)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for f in features:
            self.downs.append(DoubleConv(in_ch, f))
            in_ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)   #32 -> 64 at 8x8

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))                      # f (skip) + f (upsampled)

        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)                                                 # (B, 64, 8, 8)

        skips = skips[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]
            x = self.ups[idx + 1](torch.cat((skip, x), dim=1))

        return self.final_conv(x)


# 5. Losses
def dice_loss(logits, target, smooth=1.0):
    """
    Dice overlap between sigmoid(logits) and target, averaged over the batch.
    dice = (2 * sum(p * t) + eps) / (sum(p) + sum(t) + eps)
    Apply to only cracked walls
    """
    pred = torch.sigmoid(logits)
    intersection = (pred*target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def total_loss(mask_logits, mask_true, cracked):
    """
    BCE on every wall's mask
    Dice only where there is a crack
    """
    # crack pixels are ~1% of the image; pos_weight stops "predict nothing" being the easy answer
    pos_weight = torch.tensor(POS_WEIGHT, device=mask_logits.device)
    bce = F.binary_cross_entropy_with_logits(mask_logits, mask_true, pos_weight=pos_weight)
    loss = bce
    dice = torch.tensor(0.0)                  # default for a batch with no cracked walls
    if cracked.any():
        dice = dice_loss(mask_logits[cracked], mask_true[cracked])
        loss = loss + dice
    return loss, (bce.item(), dice.item())



# 6. Training and output
def augment(x, mask):
    """
    8x data set for training
    """
    if torch.rand(1) < 0.5:
        x, mask = x.flip(-1), mask.flip(-1)     # left-right
    if torch.rand(1) < 0.5:
        x, mask = x.flip(-2), mask.flip(-2)     # up-down
    if torch.rand(1) < 0.5:
        x, mask = x.transpose(-1, -2), mask.transpose(-1, -2)   # swap rows <-> cols

    x = x + noise_std * torch.randn_like(x)
    return x, mask


def train_one_epoch(model, loader, optimizer):
    """model.train(); for each batch: augment, forward, total_loss, backward, step.
    Returns (average loss, [average bce, average dice]) over the epoch."""
    model.train()
    total, part_sums, n = 0.0, [0.0, 0.0], 0
    for x, mask, cracked in loader:
        x, mask, cracked = x.to(device), mask.to(device), cracked.to(device)

        x, mask = augment(x, mask)

        mask_logits = model(x)
        loss, parts = total_loss(mask_logits, mask, cracked)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        b = x.size(0)                                          # walls in this batch
        total += loss.item() * b
        part_sums = [s + p * b for s, p in zip(part_sums, parts)]
        n += b
    return total / n, [s / n for s in part_sums]

def evaluate(model, loader):
    """model.eval() + torch.no_grad(); no augmentation.
    Returns (average loss, [average bce, average dice])."""
    model.eval()
    total, part_sums, n = 0.0, [0.0, 0.0], 0
    with torch.no_grad():
        for x, mask, cracked in loader:
            x, mask, cracked = x.to(device), mask.to(device), cracked.to(device)

            mask_logits = model(x)
            loss, parts = total_loss(mask_logits, mask, cracked)

            b = x.size(0)                                          # walls in this batch
            total += loss.item() * b
            part_sums = [s + p * b for s, p in zip(part_sums, parts)]
            n += b
    return total / n, [s / n for s in part_sums]

def predict_params(models, X, threshold=THRESHOLD, length_threshold=LENGTH_THRESHOLD):
    """
    Run every model in the ensemble on all walls and AVERAGE their
    crack-probability maps pixel by pixel. Then for each wall:
    averaged prob -> mask_to_params -> pixels_to_params, measured twice:
    x, y, phi from the strict `threshold` mask, and length from the looser
    `length_threshold` mask.

    Returns
        out    : (N, 4) array in PRED_NAMES order (x, y, length, phi).
                 NaN where no pixel passed the threshold.
        scores : (N,) max(prob) per wall: crack/no-crack score read off the mask.
    """
    for m in models:
        m.eval()
    out = np.full((len(X), len(PRED_NAMES)), np.nan)
    scores = np.zeros(len(X))
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size].to(device)
            # .double(): in float32, sigmoid rounds any logit above ~17 to exactly 1.0, so strong
            # cracks (and the worst false alarms) would all tie at 1.0 and no cut-off could separate them
            probs = [torch.sigmoid(m(xb)[:, 0].double()) for m in models]   # each model's guess: (B, H, W)
            prob = torch.stack(probs).mean(dim=0).cpu()       # (n_models, B, H, W) -> (B, H, W): average vote
            for j in range(len(xb)):
                i = start + j
                scores[i] = prob[j].max().item()
                m = mask_to_params(prob[j], threshold)            # strict mask: centre, angle
                if m is None:
                    continue
                x, y, length, _width, phi = pixels_to_params(*m)  # width is below pixel size: ignored
                m_len = mask_to_params(prob[j], length_threshold) # looser mask: keeps the faint ends
                if m_len is not None:
                    length = pixels_to_params(*m_len)[2]          # [2] = length
                out[i] = [x, y, length, phi]                      # PRED_NAMES order
    return out, scores


def detection_cutoff(clean_scores, fa_target):
    """
    crack_score cut-off (rule: cracked if score > cut-off) that flags at most
    fa_target of these clean walls. Sort the clean scores highest first; if we may
    flag k = floor(fa_target * n) of them, the cut-off is the (k+1)-th highest score,
    so exactly the k walls above it get flagged (fewer if scores tie).
    With k = 0 the cut-off is the highest clean score: no clean wall is flagged.
    """
    s = np.sort(clean_scores)[::-1]                  # highest first
    k = int(np.floor(fa_target * len(s)))
    return s[k]


def main():
    """
    load_runs -> split_indices -> preprocess -> build_masks
    -> DataLoaders of (X, mask, cracked) for train / val
    -> UNet, Adam -> loop train_one_epoch / evaluate, keep best val weights
    -> predict_params -> print test MAE per parameter -> write csv
    """
    images, has_crack, run_ids, params = load_runs(data_dir)
    N, _, H, W = images.shape
    idx_train, idx_val, idx_test = split_indices(has_crack)
    X, _ = preprocess(images, has_crack, idx_train)
    masks = build_masks(params, has_crack, H, W)
    n_px = masks[:, 0].sum(dim=(1, 2))                     # crack pixels in view, per wall
    for name, idx in [('train', idx_train), ('val', idx_val), ('test', idx_test)]:
        c = idx[has_crack[idx]]
        print(f'{name}: {int((n_px[c] == 0).sum())} cracked walls with no crack in view, '
              f'{int((n_px[c] < 10).sum())} with fewer than 10 crack pixels')
    cracked_train = idx_train[has_crack[idx_train]]
    diffs = []
    for i in cracked_train:
        m = masks[i, 0] > 0
        diffs.append((X[i, 0][m].mean() - X[i, 0][~m].mean()).item())
    diffs = torch.tensor(diffs)
    print(f'inside - outside (peak channel): mean {diffs.mean():+.3f}, '
      f'higher inside on {(diffs > 0).float().mean() * 100:.0f}% of {len(diffs)} walls')

    def make_loader(idx, shuffle):
        ds = TensorDataset(X[idx], masks[idx], has_crack[idx])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    train_loader = make_loader(idx_train, shuffle=True)
    val_loader = make_loader(idx_val, shuffle=False)

    in_ch = X.shape[1]                       # 3 = peak, scattered, uncollided
    models = []
    for seed in range(N_MODELS):
        path = MODEL_PATH.format(seed)
        torch.manual_seed(seed)              # different seed -> different start weights,
        np.random.seed(seed)                 # batch order, flips and noise
        model = UNet(in_ch=in_ch, features=(8, 16, 32)).to(device)

        if TRAIN:
            print(f'\n--- model {seed + 1}/{N_MODELS} (seed {seed}) ---')
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            best_val, best_epoch, best_state = float('inf'), 0, None
            for epoch in range(num_epochs):
                train_loss, tr = train_one_epoch(model, train_loader, optimizer)
                val_loss, va = evaluate(model, val_loader)
                if val_loss < best_val:                  # val_loss = bce + dice
                    best_val, best_epoch = val_loss, epoch
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

                if epoch % 10 == 0 or epoch == num_epochs - 1:
                    print(f'epoch {epoch:3d}  train {train_loss:.3f} (bce {tr[0]:.3f} dice {tr[1]:.3f})'
                        f'  val {val_loss:.3f} (bce {va[0]:.3f} dice {va[1]:.3f})')
            model.load_state_dict(best_state)
            print(f'best val (bce + dice) {best_val:.4f} at epoch {best_epoch}')
            torch.save(best_state, path)
            print('saved', path)
        else:
            state = torch.load(path, map_location=device)
            model.load_state_dict(state)
            print('loaded', path, '(no training)')
        models.append(model)

    pred, scores = predict_params(models, X)
    true = params.numpy()[:, PRED_COLS]                  # true x, y, length, phi: same order as pred
    eval_idx = (idx_val if EVAL_ON == 'val' else idx_test).numpy()   # which walls to score
    eval_cracked = eval_idx[has_crack.numpy()[eval_idx]]
    train_np = idx_train.numpy()
    train_c = train_np[has_crack.numpy()[train_np]]
    base = np.nanmean(true[train_c], axis=0)            # average of each parameter over cracked training walls
    L = PRED_NAMES.index('length')
    print(f'thresholds: {THRESHOLD} (centre, angle), {LENGTH_THRESHOLD} (length)')
    print(f'{EVAL_ON} MAE on {len(eval_cracked)} cracked walls (baseline = always guess the training average):')
    for k, name in enumerate(PRED_NAMES):
        err = np.abs(pred[eval_cracked, k] - true[eval_cracked, k])
        err0 = np.abs(base[k] - true[eval_cracked, k])
        if name == 'phi':
            err, err0 = np.minimum(err, 180 - err), np.minimum(err0, 180 - err0)
        found = ~np.isnan(err)
        mae = err[found].mean() if found.any() else float('nan')
        print(f'  {name:7s} {mae:7.3f}   baseline {err0.mean():7.3f}   (measured on {found.sum()}/{len(err)})')
    d = pred[eval_cracked, L] - true[eval_cracked, L]
    print(f'length: predicted - true = {np.nanmean(d):+.2f} cm on average '
          f'(too long on {np.mean(d[~np.isnan(d)] > 0) * 100:.0f}% of walls)')
    meas = []
    for i in eval_cracked:
        m = mask_to_params(masks[i, 0], THRESHOLD)
        meas.append(np.nan if m is None else pixels_to_params(*m)[2])   # [2] = length
    meas = np.array(meas)
    d = meas - true[eval_cracked, L]
    print(f'length measured from TRUE masks: MAE {np.nanmean(np.abs(d)):.2f}, '
          f'bias {np.nanmean(d):+.2f} cm (perfect network ceiling)')

    from sklearn.metrics import roc_auc_score
    y_eval = has_crack[eval_idx].numpy()
    print(f'crack/no-crack AUC on {EVAL_ON} (max(prob)): {roc_auc_score(y_eval, scores[eval_idx]):.3f}')

    # Detection: found cracks AND false alarms, first with the old 0.5 rule, then with
    # cut-offs chosen on the clean val walls for a set false-alarm rate.
    has_np, val_np = has_crack.numpy(), idx_val.numpy()
    eval_clean = eval_idx[~has_np[eval_idx]]
    val_clean = val_np[~has_np[val_np]]
    print(f'\ndetection on {EVAL_ON}: {len(eval_cracked)} cracked, {len(eval_clean)} clean walls')
    print(f'  max(prob) > {THRESHOLD} (old rule): found {(scores[eval_cracked] > THRESHOLD).sum()}'
          f'/{len(eval_cracked)} cracks, false alarms {(scores[eval_clean] > THRESHOLD).sum()}'
          f'/{len(eval_clean)} clean walls')
    print(f'  cut-offs chosen on the {len(val_clean)} clean VAL walls:')
    for fa in FA_TARGETS:
        t = detection_cutoff(scores[val_clean], fa)
        found = (scores[eval_cracked] > t).sum()
        false_alarms = (scores[eval_clean] > t).sum()
        print(f'    allow {fa:4.0%} false alarms: cut-off {t:.4f}  ->  found {found}/{len(eval_cracked)} '
              f'({found / len(eval_cracked):.0%}), false alarms {false_alarms}/{len(eval_clean)} '
              f'({false_alarms / len(eval_clean):.0%})')

    detect_cut = detection_cutoff(scores[val_clean], DETECT_FA)
    df = pd.DataFrame(pred, columns=PRED_NAMES)          # x, y, length, phi
    df.insert(0, 'run_id', run_ids)
    df.insert(1, 'crack_score', scores)                  # max(prob) off the mask
    df.insert(2, 'detected', scores > detect_cut)        # crack_score above the DETECT_FA cut-off
    df.to_csv(out_csv, index=False)
    print('wrote', out_csv)


if __name__ == '__main__':
    main()