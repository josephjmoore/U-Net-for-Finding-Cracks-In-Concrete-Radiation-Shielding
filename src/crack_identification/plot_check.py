"""
Plot the saved flux maps of one .npz file: a finished run OR a measurement.
The geometry is plotted too when there is a run folder to read it back from.

Usage:
    python plot_run.py measurement.npz                        maps only
    python plot_run.py measurement.npz --truth original.npz   + true crack centre from another file
    python plot_run.py measurement.npz --run-dir runs/run_00074   + geometry from that run folder
    python plot_run.py 74                                     run_00074: maps + geometry, as before
    python plot_run.py                                        measurement.npz
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

# Paths are anchored to the repo root (two levels above this file, like unet.py),
# so the script works no matter which folder it is launched from.
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"
DATA_DIR = REPO_ROOT / "data"

# Must match build_geometry / build_tallies defaults
SHIELD_Z = 30.0
SHIELD_THICKNESS = 5.0
DETECTOR_MID_Z = 38.0        # centre of the 37.5-38.5 cm detector layer
SCAN_HALF_WIDTH = 20.0
PIXELS = (1000, 1000)


def find(p, must_exist=True):
    """A relative path is looked for in the current folder, then the repo root, then data/."""
    p = Path(p)
    if p.is_absolute() or p.exists():
        return p
    for folder in (REPO_ROOT, DATA_DIR):
        if (folder / p).exists():
            return folder / p
    if must_exist:
        searched = ", ".join(dict.fromkeys(str(f) for f in (Path.cwd(), REPO_ROOT, DATA_DIR)))
        raise SystemExit(f"can't find {p}: looked in {searched}. "
                         f"Move it to one of those or give its full path in quotes.")
    return p


def load_model(run_dir):
    """Rebuild the exact model that was run, from the XML files OpenMC wrote."""
    import openmc            # only needed for the geometry, so maps work without OpenMC installed
    openmc.config["cross_sections"] = os.environ.get(
        "OPENMC_CROSS_SECTIONS", "/home/asus/nuclear_data/endfb-viii.0-hdf5/cross_sections.xml")
    if (run_dir / "model.xml").exists():
        return openmc.Model.from_model_xml(run_dir / "model.xml")
    return openmc.Model.from_xml(geometry=str(run_dir / "geometry.xml"),
                                 materials=str(run_dir / "materials.xml"),
                                 settings=str(run_dir / "settings.xml"))


def has_params(npz):
    """True if the file still has its crack parameters (a measurement may not)."""
    return "has_crack" in npz.files


def load_crack(npz):
    """Crack parameters as a dict (phi recovered from sin2phi/cos2phi), or None."""
    if not bool(npz["has_crack"]):
        return None
    crack = dict(zip([str(l) for l in npz["param_labels"]], npz["params"].astype(float)))
    crack["phi"] = 0.5 * np.arctan2(crack["sin2phi"], crack["cos2phi"]) % np.pi
    return crack


def cell_colours(geometry):
    colours = {}
    for cell in geometry.get_all_cells().values():
        colours[cell] = {"crack": "red", "shield": "grey"}.get(cell.name, "lightblue")
    return colours


def plot_geometry(model, crack, name):
    root = model.geometry.root_universe
    colours = cell_colours(model.geometry)
    shield_back = SHIELD_Z + SHIELD_THICKNESS / 2

    if crack is None:
        views = [("xy, whole world (z = %.1f cm)" % SHIELD_Z,
                  dict(basis="xy", origin=(0, 0, SHIELD_Z), width=(120, 120)))]
    else:
        z_slice = shield_back - min(crack["depth"], SHIELD_THICKNESS) / 2
        zoom = crack["length"] + 10.0
        centre = (crack["x"], crack["y"])
        views = [
            ("xy, whole world (z = %.1f cm)" % z_slice,
             dict(basis="xy", origin=(0, 0, z_slice), width=(120, 120))),
            ("xy, zoomed on crack",
             dict(basis="xy", origin=(*centre, z_slice), width=(zoom, zoom))),
            ("xz through crack (y = %.1f cm)" % crack["y"],
             dict(basis="xz", origin=(*centre, SHIELD_Z + 4.0), width=(zoom, 30))),
        ]

    fig, axes = plt.subplots(1, len(views), figsize=(5.5 * len(views), 5.5), squeeze=False)
    for ax, (title, view) in zip(axes[0], views):
        root.plot(pixels=PIXELS, color_by="cell", colors=colours, axes=ax, **view)
        ax.set_title(title)
    fig.suptitle(f"{name}: grey = concrete, red = crack, blue = air")
    fig.tight_layout()
    out = f"{name}_geometry.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


def plot_maps(npz, crack, name, known=True):
    """known=False: the file has no crack parameters, so say 'unknown' rather than 'no crack'."""
    image = npz["image"]
    peak, scattered = image[0], image[1]

    extent = [-SCAN_HALF_WIDTH, SCAN_HALF_WIDTH, -SCAN_HALF_WIDTH, SCAN_HALF_WIDTH]
    panels = [
        ("Peak window (600-662 keV)", peak, True, "photons cm$^{-2}$ s$^{-1}$"),
        ("Scattered (< 600 keV)", scattered, True, "photons cm$^{-2}$ s$^{-1}$"),
    ]
    peak_rel_err = None
    if "image_std" in npz.files:                       # a measurement file may not have it
        with np.errstate(divide="ignore", invalid="ignore"):
            peak_rel_err = np.where(peak > 0, npz["image_std"][0] / peak, np.nan)
        panels.append(("Peak relative error", 100 * peak_rel_err, False, "%"))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.7 * len(panels), 5.2), squeeze=False)
    for ax, (title, data, log, units) in zip(axes[0], panels):
        shown = np.where(data > 0, data, np.nan) if log else data
        norm = LogNorm(np.nanmin(shown), np.nanmax(shown)) if log else None
        # Arrays are [iy, ix]; origin="lower" puts +y at the top, +x to the right
        im = ax.imshow(shown, origin="lower", extent=extent, cmap="Blues", norm=norm)
        fig.colorbar(im, ax=ax, shrink=0.85, label=units)
        ax.set_title(title)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")

        if crack is not None:
            # Point source: the crack's image on the detector is magnified
            # by detector_z / back-face_z
            f = DETECTOR_MID_Z / (SHIELD_Z + SHIELD_THICKNESS / 2)
            ax.plot(crack["x"] * f, crack["y"] * f, marker="x", color="black",
                    markersize=9, linestyle="none", label="crack centre (projected)")
            ax.legend(loc="upper right", fontsize=8, frameon=True)

    if not known:
        sub = "crack parameters unknown"
    elif crack is None:
        sub = "no crack"
    else:
        sub = ("crack at (%.1f, %.1f) cm, width %.2f cm, length %.1f cm, depth %.1f cm, "
               "phi %.0f deg" % (crack["x"], crack["y"], crack["width"], crack["length"],
                                 crack["depth"], np.degrees(crack["phi"])))
    fig.suptitle(f"{name}: {sub}")
    fig.tight_layout()
    out = f"{name}_maps.png"
    fig.savefig(out, dpi=150)
    print("saved", out)
    if peak_rel_err is not None:
        print("peak relative error: median %.1f%%, max %.1f%%"
              % (100 * np.nanmedian(peak_rel_err), 100 * np.nanmax(peak_rel_err)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", nargs="?", default="measurement.npz",
                        help="an .npz file, or a run number such as 74 (default: measurement.npz)")
    parser.add_argument("--truth", help=".npz that still has the crack parameters, for the crack centre")
    parser.add_argument("--run-dir", help="run folder with model.xml, to plot the geometry too")
    args = parser.parse_args()

    if args.file.isdigit():                                    # a run number: as before
        name = f"run_{int(args.file):05d}"
        path = DATA_DIR / f"{name}.npz"
        run_dir = find(args.run_dir, must_exist=False) if args.run_dir else RUNS_DIR / name
    else:                                                      # any .npz, e.g. measurement.npz
        path = find(args.file)
        name = path.stem
        run_dir = find(args.run_dir, must_exist=False) if args.run_dir else None

    npz = np.load(path)
    truth = np.load(find(args.truth)) if args.truth else npz         # where the crack parameters come from
    known = has_params(truth)
    crack = load_crack(truth) if known else None

    if run_dir is not None and run_dir.exists():
        plot_geometry(load_model(run_dir), crack, name)
    elif run_dir is not None:
        print(f"no run folder at {run_dir}: skipping the geometry plot")
    plot_maps(npz, crack, name, known)
    plt.show()


if __name__ == "__main__":
    main()