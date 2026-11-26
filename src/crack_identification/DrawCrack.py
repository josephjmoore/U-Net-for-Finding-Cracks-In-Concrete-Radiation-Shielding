"""
Draw the crack the U-Net ensemble finds in ONE flux measurement, with its
uncertainty, and the true crack for comparison. 
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch

from unet import (UNet, load_runs, split_indices, preprocess, predict_params,
                  PRED_NAMES, PIXEL_SIZE_CM, ORIGIN_PX, THRESHOLD, LENGTH_THRESHOLD,
                  N_MODELS, MODEL_PATH, data_dir, device)

def load_measurement(path):
    """
    Read one flux measurement
    Returns a (3, H, W) float32 tensor: peak, scattered, uncollided.
    """
    with np.load(path) as f:
        image = torch.tensor(f['image'], dtype=torch.float32)               # (2, H, W): peak, scattered
        unc = torch.tensor(f['uncollided'], dtype=torch.float32)[None]      # (1, H, W)
    return torch.cat([image, unc], dim=0)                                   # (3, H, W)

def read_known(path):
    """
    True crack from an .npz that still has its parameters.
    Returns {'x', 'y', 'length', 'phi'} (cm, cm, cm, degrees in [0, 180)),
    or None if f['has_crack'] is False.
    """
    with np.load(path) as f:
        if not bool(f['has_crack']):
            return None                                        # clean wall: nothing to draw
        p = dict(zip(f['param_labels'], f['params']))          
    phi = np.degrees(np.arctan2(p['sin2phi'], p['cos2phi'])) / 2 % 180
    return {'x': float(p['x']), 'y': float(p['y']), 'length': float(p['length']), 'phi': float(phi)}

def normalise_like_training(image, train_dir=data_dir):
    """
    Turn a raw (3, H, W) measurement into network input: (1, 3, H, W) 
    using the training set's clean-wall reference and channel mean/std.
    """
    images, has_crack, _, _ = load_runs(train_dir)
    idx_train, _, _ = split_indices(has_crack)                 # identical split to training
    all_images = torch.cat([images, image[None]])              # new image is the last one
    all_crack = torch.cat([has_crack, torch.tensor([False])])  # its label is never used
    X, _ = preprocess(all_images, all_crack, idx_train)
    return X[-1:]                                              # (1, 3, H, W)


def load_ensemble(in_ch=3):
    """
    Load the N_MODELS saved U-Nets (MODEL_PATH.format(seed) for seed 0..N_MODELS-1).
    Returns a list of models, each in eval mode, on `device`.
    """
    models = []
    for seed in range(N_MODELS):
            model = UNet(in_ch=in_ch, features=(8, 16, 32)).to(device)
            state = torch.load(MODEL_PATH.format(seed), map_location=device)
            model.load_state_dict(state)
            model.eval()
            models.append(model)
    return models

def probability_maps(models, X):
    """
    Each model's crack-probability map for the single input X (1, 3, H, W).
    Returns a numpy array (n_models, H, W) of values in [0, 1].
    """
    with torch.no_grad():
        return np.stack([torch.sigmoid(m(X.to(device)))[0, 0].cpu().numpy() for m in models])

def as_dict(row):
    """
    One row of predict_params output -> {'x', 'y', 'length', 'phi'}, or None
    if no crack was found
    """
    return None if np.isnan(row[0]) else dict(zip(PRED_NAMES, row))

def predict_crack(models, X):
    """
    The ensemble's crack AND each single model's crack, plus the probability maps.

    Returns a dict:
        'ensemble'  : as_dict of predict_params(models, X)        (all 5 averaged)
        'members'   : [as_dict of predict_params([m], X) for each model m]
        'score'     : the crack score predict_params returns for this wall
        'prob_mean' : (H, W) mean of probability_maps over models
        'prob_std'  : (H, W) std of probability_maps over models  (disagreement)

    """
    out, scores = predict_params(models, X)          
    members = []
    for m in models:
        member_out, _ = predict_params([m], X)        
        members.append(as_dict(member_out[0]))        
    probs = probability_maps(models, X)               # (5, H, W)
    return {'ensemble': as_dict(out[0]),
            'members': members,
            'score': float(scores[0]),
            'prob_mean': probs.mean(axis=0),          
            'prob_std': probs.std(axis=0)}            
        
def spread(members):
    """
    Uncertainty: mean and standard deviation of each parameter over the models
    that found a crack (skip the Nones).

    Returns {'n_found': int, 'x': (mean, std), 'y': ..., 'length': ..., 'phi': ...}
    (only 'n_found' if no model found a crack).
    """
    found = [m for m in members if m is not None]
    out = {'n_found': len(found)}
    if not found:
        return out
    for name in ('x', 'y', 'length'):
        v = np.array([m[name] for m in found])
        out[name] = (v.mean(), v.std())
    phi = np.radians([m['phi'] for m in found])
    mean_phi = np.degrees(np.arctan2(np.sin(2 * phi).mean(), np.cos(2 * phi).mean()) / 2) % 180 # because angle is orientation
    dev = (np.degrees(phi) - mean_phi + 90) % 180 - 90
    out['phi'] = (mean_phi, dev.std())
    return out

def crack_segment(c):
    """
    A crack as a line segment in wall cm.

    c: {'x', 'y', 'length', 'phi'}.  Returns ([x_start, x_end], [y_start, y_end]),
    ready for ax.plot(xs, ys).
    """
    dx = np.cos(np.radians(c['phi'])) * c['length'] / 2
    dy = np.sin(np.radians(c['phi'])) * c['length'] / 2
    return [c['x'] - dx, c['x'] + dx], [c['y'] - dy, c['y'] + dy]


def image_extent(H=64, W=64):
    """
    [x_left, x_right, y_bottom, y_top] in wall cm, for imshow(..., extent=...),
    so the axes read in cm and line up with crack_segment.
    """
    x_left = (-0.5 - ORIGIN_PX[0]) * PIXEL_SIZE_CM
    x_right = (W - 0.5 - ORIGIN_PX[0]) * PIXEL_SIZE_CM
    y_bottom = (-0.5 - ORIGIN_PX[1]) * PIXEL_SIZE_CM
    y_top = (H - 0.5 - ORIGIN_PX[0]) * PIXEL_SIZE_CM
    return x_left, x_right, y_bottom, y_top


def draw_cracks(ax, result, truth=None):
    """
    Overlay on one axis:
      each model's crack   thin, semi-transparent 
      the ensemble crack   thick 
      the true crack       dashed; white with a black outline 
    """
    for k, m in enumerate(result['members']):
        if m is not None:
            ax.plot(*crack_segment(m), color='tab:orange', lw=1, alpha=0.6,
                    label='single models' if k == 0 else None)
    if result['ensemble'] is not None:
        ax.plot(*crack_segment(result['ensemble']), color='tab:pink', lw=3, label='ensemble')
    if truth is not None:                    # white dashes with a black outline: visible on any colour
        ax.plot(*crack_segment(truth), color='white', lw=2, ls='--', label='true crack',
                path_effects=[pe.Stroke(linewidth=3.5, foreground='black'), pe.Normal()])

def plot_prediction(X, result, truth=None, save_path='crack_prediction.png'):
    """
    One figure, three panels side by side, all with extent=image_extent(...):
      1. X[0, 0]: the measurement (peak channel, standardised log flux / clean wall)
      2. result['prob_mean'], vmin=0, vmax=1, with ax.contour lines at
         LENGTH_THRESHOLD and THRESHOLD
      3. result['prob_std']: where the models disagree
    """
    H, W = X.shape[-2:]
    ext = image_extent(H, W)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    # origin='lower': image row 0 (most negative y) at the bottom, so +y points up
    panels = [(X[0, 0].numpy(), 'measurement: peak flux vs clean wall', 'magma', None),
              (result['prob_mean'], 'crack probability (ensemble mean)', 'viridis', (0, 1)),
              (result['prob_std'], 'model disagreement (std)', 'cividis', None)]
    for ax, (img, title, cmap, lims) in zip(axes, panels):
        im = ax.imshow(img, origin='lower', extent=ext, cmap=cmap,
                       vmin=None if lims is None else lims[0], vmax=None if lims is None else lims[1])
        fig.colorbar(im, ax=ax, fraction=0.046)
        draw_cracks(ax, result, truth)
        ax.set_xlim(ext[0], ext[1])                  # keep the view on the image even if a
        ax.set_ylim(ext[2], ext[3])                  # predicted line runs past its edge
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('y (cm)')
    xs = np.linspace(ext[0], ext[1], W)
    ys = np.linspace(ext[2], ext[3], H)
    axes[1].contour(xs, ys, result['prob_mean'], levels=[LENGTH_THRESHOLD, THRESHOLD],
                    colors=['white', 'white'], linewidths=[0.6, 1.2], linestyles=[':', '-'])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(summary_title(result, truth), fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 0.93))         # room for the legend below and the title above
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9, frameon=False)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path

def summary_title(result, truth):
    s = spread(result['members'])
    if result['ensemble'] is None:
        return f'no crack found (max probability {result["score"]:.2f} < {THRESHOLD})'
    e = result['ensemble']
    parts = [f'{n} = {e[n]:.1f} ± {s[n][1]:.1f}' if n in s else f'{n} = {e[n]:.1f}' for n in PRED_NAMES]
    title = f'prediction (crack score {result["score"]:.2f}, {s["n_found"]}/{len(result["members"])} models): ' + ', '.join(parts)
    if truth is not None:
        title += '\ntrue: ' + ', '.join(f'{n} = {truth[n]:.1f}' for n in PRED_NAMES)
    return title

def print_report(result, truth=None):
    """
    Prints: crack score, how many models found a crack, a table with one
    row per parameter: ensemble value, ± spread, and if truth is given, true value
    and error.
    """
    s = spread(result['members'])
    print(f'crack score (max probability): {result["score"]:.3f}   '
          f'models finding a crack: {s["n_found"]}/{len(result["members"])}')
    if result['ensemble'] is None:
        print('no crack found by the ensemble')
        return
    unit = {'x': 'cm', 'y': 'cm', 'length': 'cm', 'phi': 'deg'}
    print(f'  {"":7s} {"ensemble":>9s} {"± spread":>9s}' + (f' {"true":>8s} {"error":>8s}' if truth else ''))
    for n in PRED_NAMES:
        e = result['ensemble'][n]
        row = f'  {n:7s} {e:9.2f} {s[n][1] if n in s else float("nan"):9.2f}'
        if truth:
            err = e - truth[n]
            if n == 'phi':
                err = (err + 90) % 180 - 90
            row += f' {truth[n]:8.2f} {err:+8.2f}'
        print(row + f'  {unit[n]}')


MEASUREMENT_FILE = os.path.join(data_dir, 'measurement5.npz')   # data_dir comes from DATA_DIR
OUT_PNG = 'crack_prediction5.png'

def main():
    image = load_measurement(MEASUREMENT_FILE)
    X = normalise_like_training(image)
    models = load_ensemble(in_ch=X.shape[1])
    result = predict_crack(models, X)
    truth = read_known(MEASUREMENT_FILE)      
    print_report(result, truth)
    print('saved', plot_prediction(X, result, truth, OUT_PNG))

if __name__ == '__main__':
        main()