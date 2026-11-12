import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 100
batch_size = 32
learning_rate = 0.001     # keep at 0.001 for Adam; 0.01-0.1 is too high
noise_std = 0.2           # fresh noise added to each training batch (standardised units)

data_dir = os.environ.get('DATA_DIR', 'data')

images, has_crack = [], []
for i in range(0, 251):                       # run_00000 .. run_00250
    run_i = os.path.join(data_dir, f'run_{i:05d}.npz')
    if not os.path.exists(run_i):
        print(f'skipping missing {run_i}')
        continue
    with np.load(run_i) as f:
        images.append(torch.tensor(f['image'], dtype=torch.float32))    # (2, H, W) flux
        has_crack.append(bool(f['has_crack']))

images = torch.stack(images)                  # (N, 2, H, W)
has_crack = torch.tensor(has_crack)           # (N,) bool
N, _, H, W = images.shape
print(f'{N} runs loaded, {int(has_crack.sum())} cracked, image size {H}x{W}')


# Split by index into train / validation / test (70 / 15 / 15), keeping the cracked/clean ratio the same in each
idx = np.arange(N)
idx_rest, idx_test = train_test_split(
    idx, test_size=0.15, random_state=1234, stratify=has_crack.numpy())
idx_train, idx_val = train_test_split(
    idx_rest, test_size=0.15 / 0.85, random_state=1234,
    stratify=has_crack.numpy()[idx_rest])
idx_train, idx_val, idx_test = (torch.as_tensor(a) for a in (idx_train, idx_val, idx_test))


# Inputs: log(flux / clean-wall reference), per-channel standardisation
clean_train = idx_train[~has_crack[idx_train]]
assert len(clean_train) > 0, 'need at least one clean wall in the training set'
ref = images[clean_train].mean(dim=0)                       # (2, H, W), clean-wall average

ratio = (images / ref).clamp_min(1e-3)                      # guard against empty pixels
X = torch.log(ratio)                                        # ~0 on clean walls, >0 at cracks
n_clamped = int((ratio <= 1e-3).sum())
# Real crack signal in 5 cm of concrete is at most ~2.4x (log 0.88); cap anything
# beyond +-1 so a few near-empty pixels can't become huge outliers
X = X.clamp(-1.0, 1.0)

# One mean and std per channel, from the training set only
ch_mean = X[idx_train].mean(dim=(0, 2, 3), keepdim=True)    # (1, 2, 1, 1)
ch_std = X[idx_train].std(dim=(0, 2, 3), keepdim=True)
X = (X - ch_mean) / (ch_std + 1e-8)                         # (N, 2, H, W): peak, scattered
print(f'input check: min {X.min().item():.1f}, max {X.max().item():.1f}, '
      f'clamped pixels {n_clamped}, NaN/inf {bool(torch.isnan(X).any() or torch.isinf(X).any())}')


# Target: crack present as a float (BCE loss needs floats)
y = has_crack.float()                                       # (N,) 1.0 / 0.0


def make_dataset(ix):
    return TensorDataset(X[ix], y[ix])

train_loader = DataLoader(make_dataset(idx_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(make_dataset(idx_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(make_dataset(idx_test), batch_size=batch_size, shuffle=False)

print('X', tuple(X.shape), '| y', tuple(y.shape))
print(f'train {len(idx_train)}, val {len(idx_val)}, test {len(idx_test)}')

# Baseline: brightest lightly-smoothed spot in the peak channel, no learning.
# ~0.5 means a simple score can't tell cracked from clean walls (data problem);
# clearly above 0.5 means the signal is there for the CNN to find.
baseline_score = F.avg_pool2d(X[:, :1], 3, stride=1).amax(dim=(1, 2, 3))
print(f'baseline AUC {roc_auc_score(y.numpy(), baseline_score.numpy()):.3f}')


class ConvNet(nn.Module):
    """
    Two conv blocks, then global max pooling: each of the 64 feature maps reports
    its strongest response anywhere in the image ("is there a crack-like streak?").
    ~20k weights instead of ~1.6M, so it can't simply memorise 139 walls.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)      # 2 input channels -> 32 maps
        self.bn1 = nn.BatchNorm2d(32)                     # keeps activations well scaled
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)     # input must match conv1's 32 outputs
        self.bn2 = nn.BatchNorm2d(64)
        self.gpool = nn.AdaptiveMaxPool2d(1)              # strongest response per feature map
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 1)                        # one raw score (logit) per wall

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))    # (B, 32, 31, 31)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))    # (B, 64, 14, 14)
        x = torch.flatten(self.gpool(x), 1)               # (B, 64)
        return self.fc(self.dropout(x))                   # (B, 1)


model = ConvNet().to(device)
print(f'model has {sum(p.numel() for p in model.parameters()):,} weights')

# BCEWithLogitsLoss applies the sigmoid itself, so the model outputs raw scores
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)


def evaluate(loader):
    """Average loss and accuracy over a loader. A logit > 0 means probability > 0.5."""
    model.eval()
    total_loss, n_correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb).squeeze(1)                 # (B,)
            total_loss += criterion(logits, yb).item() * len(yb)
            n_correct += ((logits > 0).float() == yb).sum().item()
            n += len(yb)
    return total_loss / n, n_correct / n


# Training, keeping the weights from the epoch with the lowest validation loss
best_val_loss, best_state, best_epoch = float('inf'), None, 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for xb, yb in train_loader:                           # loader gives (images, labels)
        xb, yb = xb.to(device), yb.to(device)

        # Augmentation (training only). Flips don't change whether a wall is cracked,
        # and fresh noise stops the network memorising each wall's fixed noise pattern.
        if torch.rand(1).item() < 0.5:
            xb = xb.flip(3)                               # left-right
        if torch.rand(1).item() < 0.5:
            xb = xb.flip(2)                               # up-down
        xb = xb + noise_std * torch.randn_like(xb)

        # Forward pass
        logits = model(xb).squeeze(1)                     # (B, 1) -> (B,) to match yb
        loss = criterion(logits, yb)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(yb)

    train_loss = running_loss / len(train_loader.dataset)
    val_loss, val_acc = evaluate(val_loader)
    if val_loss < best_val_loss:
        best_val_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1
    print(f'Epoch [{epoch + 1}/{num_epochs}]  train loss {train_loss:.4f}  '
          f'val loss {val_loss:.4f}  val acc {100 * val_acc:.1f}%')

model.load_state_dict(best_state)
print(f'restored weights from epoch {best_epoch} (lowest validation loss)')


# Test: overall accuracy, plus accuracy on clean and cracked walls separately
model.eval()
probs, labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb.to(device)).squeeze(1).cpu()
        probs.append(torch.sigmoid(logits))
        labels.append(yb)
probs, labels = torch.cat(probs), torch.cat(labels)
preds = (probs > 0.5).float()

acc = 100.0 * (preds == labels).float().mean().item()
print(f'Test accuracy: {acc:.1f} % ({len(labels)} walls)')
for cls, name in [(0.0, 'clean walls (correctly called clean)'),
                  (1.0, 'cracked walls (detection rate)')]:
    mask = labels == cls
    if mask.any():
        cls_acc = 100.0 * (preds[mask] == labels[mask]).float().mean().item()
        print(f'  {name}: {cls_acc:.1f} % of {int(mask.sum())}')

# Ranking quality, independent of the 0.5 threshold
print(f'mean probability: clean {probs[labels == 0].mean().item():.3f}, '
      f'cracked {probs[labels == 1].mean().item():.3f}')
print(f'CNN test AUC {roc_auc_score(labels.numpy(), probs.numpy()):.3f}')
print(f'baseline AUC on test walls '
      f'{roc_auc_score(y[idx_test].numpy(), baseline_score[idx_test].numpy()):.3f}')