import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Geometry
PIXEL_SIZE_CM = 40/64      # cm per pixel = detector width / 64
ORIGIN_PX = (31.5, 31.5)    # (col, row) pixel position of x = 0, y = 0
Y_UP = True                 # True if +y in cm points towards row 0 (up the image)
MIN_WIDTH_PX = 2.0          # cracks thinner than this are drawn this wide

num_epochs = 150
batch_size = 16
learning_rate = 0.001
noise_std = 0.2
POS_WEIGHT = 20.0           # BCE weight on crack pixels (they are ~1% of each image)
data_dir = os.environ.get('DATA_DIR', 'data')
out_csv = os.environ.get('OUT_CSV', 'crack_params_unet.csv')
PARAM_NAMES = ['x', 'y', 'width', 'length', 'depth', 'phi']

def load_runs(data_dir):
    """
    Load every run_XXXXX.npz in data_dir.
    Returns
        run_ids   : list of str, e.g. 'run_00074'
        images    : (N, 2, H, W) float32 tensor
        has_crack : (N,) bool tensor
        params    : (N, 6) float tensor, columns PARAM_NAMES, NaN for clean walls.
                    phi in degrees [0, 180) = atan2(sin2phi, cos2phi) / 2.
    """
    images, has_crack, params, run_ids = [], [], [], []
    for i in range(0, 251):                       # run_00000 .. run_00250
        run_i = os.path.join(data_dir, f'run_{i:05d}.npz')
        if not os.path.exists(run_i):
            print(f'skipping missing {run_i}')
            continue
        with np.load(run_i) as f:
            images.append(torch.tensor(f['image'], dtype=torch.float32))    # (2, H, W) flux
            has_crack.append(bool(f['has_crack']))
            params.append(torch.tensor(f['params'], dtype=torch.float32))   # (7,), all NaN if clean
            run_ids.append(f'run_{i:05d}')

    images = torch.stack(images) #(N, 2, H, W)
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

    Returns X: (N, 2, H, W) float tensor.
    """
    clean_train = idx_train[~has_crack[idx_train]]
    assert len(clean_train) > 0, 'need at least one clean wall in the training set'
    ref = images[clean_train].mean(dim=0)  # (2, H, W), clean-wall average
    ratio = (images / ref).clamp_min(1e-3)                      # guard against empty pixels
    X = torch.log(ratio)                                        # ~0 on clean walls, >0 at cracks
    n_clamped = int((ratio <= 1e-3).sum())
    # Real crack signal in 5 cm of concrete is at most ~2.4x (log 0.88); cap anything
    # beyond +-1 so a few near-empty pixels can't become huge outliers
    X = X.clamp(-1.0, 1.0)
    ch_mean = X[idx_train].mean(dim=(0, 2, 3), keepdim=True)    # (1, 2, 1, 1)
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
    if Y_UP:
        row = ORIGIN_PX[1] - y / PIXEL_SIZE_CM      
    else:
        row = ORIGIN_PX[1] + y / PIXEL_SIZE_CM
    length_px = length / PIXEL_SIZE_CM
    width_px = max(width / PIXEL_SIZE_CM, MIN_WIDTH_PX)
    phi_px_rad = np.deg2rad(phi_deg)
    if Y_UP:
        phi_px_rad = -phi_px_rad              

    return col, row, length_px, width_px, phi_px_rad


def pixels_to_params(col, row, length_px, width_px, phi_px_rad):
    """
    inverse of params_to_pixels 
    Returns (x, y, length, width, phi_deg) in cm / degrees, phi_deg in [0, 180]).
    """
    x = (col - ORIGIN_PX[0]) * PIXEL_SIZE_CM
    if Y_UP:
        y = (ORIGIN_PX[1] - row) * PIXEL_SIZE_CM
    else:
        y = (row - ORIGIN_PX[1]) * PIXEL_SIZE_CM
    length = length_px * PIXEL_SIZE_CM
    width = width_px * PIXEL_SIZE_CM
    if Y_UP:
        phi_px_rad = -phi_px_rad
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
    rows, cols = torch.where(prob > threshold)    # integer indices of crack pixels
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
    forward(x) returns (mask_logits (B, 1, H, W), depth (B,)).
    """
    def __init__(self, in_ch=2, out_ch=1, features=(16, 32, 64)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for f in features:
            self.downs.append(DoubleConv(in_ch, f))
            in_ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)   # 64 -> 128 at 8x8

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))                      # f (skip) + f (upsampled)

        self.final_conv = nn.Conv2d(features[0], out_ch, kernel_size=1)
        self.depth_head = nn.Linear(features[-1] * 2, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        depth = self.depth_head(x.mean(dim=(2, 3))).squeeze(1)        # (B,)

        skips = skips[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]
            x = self.ups[idx + 1](torch.cat((skip, x), dim=1))

        return self.final_conv(x), depth


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


def total_loss(mask_logits, depth_pred, mask_true, depth_true, cracked):
    """
    BCE on every wall
    Dice and depth only where there is a crack
    """
    # crack pixels are ~1% of the image; pos_weight stops "predict nothing" being the easy answer
    pos_weight = torch.tensor(POS_WEIGHT, device=mask_logits.device)
    loss = F.binary_cross_entropy_with_logits(mask_logits, mask_true, pos_weight=pos_weight)
    if cracked.any():
        loss = loss + dice_loss(mask_logits[cracked], mask_true[cracked])
        loss = loss + F.mse_loss(depth_pred[cracked], depth_true[cracked])
    return loss


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
    Returns the average loss over the epoch."""
    model.train()
    total, n = 0.0, 0
    for x, mask, depth, cracked in loader:
        x, mask, depth, cracked = x.to(device), mask.to(device), depth.to(device), cracked.to(device)

        x, mask = augment(x, mask)

        mask_logits, depth_pred = model(x)
        loss = total_loss(mask_logits, depth_pred, mask, depth, cracked)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n

def evaluate(model, loader):
    """model.eval() + torch.no_grad(); returns the average total_loss (no augmentation)."""
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, mask, depth, cracked in loader:
            x, mask, depth, cracked = x.to(device), mask.to(device), depth.to(device), cracked.to(device)

            mask_logits, depth_pred = model(x)
            loss = total_loss(mask_logits, depth_pred, mask, depth, cracked)

            total += loss.item() * x.size(0)
            n += x.size(0)
    return total / n

def predict_params(model, X, depth_mean, depth_std, threshold=0.5):
    """
    Run the model on all walls; for each: sigmoid -> mask_to_params ->
    pixels_to_params, plus the depth output (un-standardised).

    Returns
        out    : (N, 6) array in PARAM_NAMES order. Geometry is NaN where no pixel
                 passed the threshold; depth is always filled in.
        scores : (N,) max(prob) per wall, a crack/no-crack score.
    """
    model.eval()
    out = np.full((len(X), 6), np.nan)
    scores = np.zeros(len(X))
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            xb = X[start:start + batch_size].to(device)
            logits, depth = model(xb)
            prob = torch.sigmoid(logits[:, 0]).cpu()          # (B, H, W)
            depth = depth.cpu() * depth_std + depth_mean      # undo standardisation
            for j in range(len(xb)):
                i = start + j
                scores[i] = prob[j].max().item()
                out[i, 4] = depth[j].item()
                m = mask_to_params(prob[j], threshold)
                if m is None:
                    continue
                x, y, length, width, phi = pixels_to_params(*m)
                out[i, [0, 1, 2, 3, 5]] = [x, y, width, length, phi]   # PARAM_NAMES order
    return out, scores


def main():
    """
    load_runs -> split_indices -> preprocess -> build_masks
    -> DataLoaders of (X, mask, depth_std, cracked) for train / val
    -> UNet, Adam -> loop train_one_epoch / evaluate, keep best val weights
    -> predict_params -> print test MAE per parameter -> write csv
    """
    images, has_crack, run_ids, params = load_runs(data_dir)
    N, _, H, W = images.shape
    idx_train, idx_val, idx_test = split_indices(has_crack)
    X, _ = preprocess(images, has_crack, idx_train)
    masks = build_masks(params, has_crack, H, W)

    # standardise depth with cracked TRAINING walls only (NaN stays NaN on clean walls)
    depth = params[:, 4]
    train_cracked = idx_train[has_crack[idx_train]]
    depth_mean, depth_std = depth[train_cracked].mean(), depth[train_cracked].std()
    depth_s = (depth - depth_mean) / depth_std

    def make_loader(idx, shuffle):
        ds = TensorDataset(X[idx], masks[idx], depth_s[idx], has_crack[idx])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    train_loader = make_loader(idx_train, shuffle=True)
    val_loader = make_loader(idx_val, shuffle=False)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val, best_state, best_epoch = float('inf'), None, -1
    for epoch in range(num_epochs):
        tr = train_one_epoch(model, train_loader, optimizer)
        va = evaluate(model, val_loader)
        if va < best_val:
            best_val, best_epoch = va, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(f'epoch {epoch:3d}  train {tr:.4f}  val {va:.4f}')
    print(f'best val {best_val:.4f} at epoch {best_epoch}')
    model.load_state_dict(best_state)

    pred, scores = predict_params(model, X, depth_mean.item(), depth_std.item())
    true = params.numpy()
    test = idx_test.numpy()
    test_cracked = test[has_crack.numpy()[test]]
    print(f'test MAE on {len(test_cracked)} cracked walls:')
    for k, name in enumerate(PARAM_NAMES):
        err = np.abs(pred[test_cracked, k] - true[test_cracked, k])
        if name == 'phi':
            err = np.minimum(err, 180 - err)                   # 179 deg and 1 deg are 2 deg apart
        found = ~np.isnan(err)
        mae = err[found].mean() if found.any() else float('nan')
        print(f'  {name:7s} {mae:7.3f}   (measured on {found.sum()}/{len(err)})')

    from sklearn.metrics import roc_auc_score
    print('crack/no-crack AUC from max(prob):',
          round(roc_auc_score(has_crack[test].numpy(), scores[test]), 3))

    df = pd.DataFrame(pred, columns=PARAM_NAMES)
    df.insert(0, 'run_id', run_ids)
    df.insert(1, 'crack_score', scores)
    df.to_csv(out_csv, index=False)
    print('wrote', out_csv)


if __name__ == '__main__':
    main()
