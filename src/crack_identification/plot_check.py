"""
Plot one finished run: its geometry (read back from the run folder) and its
saved flux maps (read from the .npz).

Usage:  python plot_run.py 3        (plots run_00003)
        python plot_run.py          (plots run_00000)
"""
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmc
from matplotlib.colors import LogNorm

RUNS_DIR = "runs"
DATA_DIR = "data"

# Must match build_geometry / build_tallies defaults
SHIELD_Z = 30.0
SHIELD_THICKNESS = 5.0
DETECTOR_MID_Z = 38.0        # centre of the 37.5-38.5 cm detector layer
SCAN_HALF_WIDTH = 20.0
PIXELS = (1000, 1000)

openmc.config["cross_sections"] = os.environ.get(
    "OPENMC_CROSS_SECTIONS", "/home/asus/nuclear_data/endfb-viii.0-hdf5/cross_sections.xml")


def load_model(run_dir):
    """Rebuild the exact model that was run, from the XML files OpenMC wrote."""
    if (run_dir / "model.xml").exists():
        return openmc.Model.from_model_xml(run_dir / "model.xml")
    return openmc.Model.from_xml(geometry=str(run_dir / "geometry.xml"),
                                 materials=str(run_dir / "materials.xml"),
                                 settings=str(run_dir / "settings.xml"))


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


def plot_geometry(model, crack, run_id):
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
    fig.suptitle(f"run {run_id}: grey = concrete, red = crack, blue = air")
    fig.tight_layout()
    out = f"run_{run_id:05d}_geometry.png"
    fig.savefig(out, dpi=150)
    print("saved", out)


def plot_maps(npz, crack, run_id):
    image, image_std = npz["image"], npz["image_std"]
    peak, scattered = image[0], image[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        peak_rel_err = np.where(peak > 0, image_std[0] / peak, np.nan)

    extent = [-SCAN_HALF_WIDTH, SCAN_HALF_WIDTH, -SCAN_HALF_WIDTH, SCAN_HALF_WIDTH]
    panels = [
        ("Peak window (600-662 keV)", peak, True, "photons cm$^{-2}$ s$^{-1}$"),
        ("Scattered (< 600 keV)", scattered, True, "photons cm$^{-2}$ s$^{-1}$"),
        ("Peak relative error", 100 * peak_rel_err, False, "%"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, (title, data, log, units) in zip(axes, panels):
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

    if crack is None:
        sub = "no crack"
    else:
        sub = ("crack at (%.1f, %.1f) cm, width %.2f cm, length %.1f cm, depth %.1f cm, "
               "phi %.0f deg" % (crack["x"], crack["y"], crack["width"], crack["length"],
                                 crack["depth"], np.degrees(crack["phi"])))
    fig.suptitle(f"run {run_id}: {sub}")
    fig.tight_layout()
    out = f"run_{run_id:05d}_maps.png"
    fig.savefig(out, dpi=150)
    print("saved", out)
    print("peak relative error: median %.1f%%, max %.1f%%"
          % (100 * np.nanmedian(peak_rel_err), 100 * np.nanmax(peak_rel_err)))


def main():
    run_id = 74
    run_name = f"run_{run_id:05d}"
    run_dir = Path(RUNS_DIR) / run_name
    npz = np.load(Path(DATA_DIR) / f"{run_name}.npz")
    crack = load_crack(npz)

    plot_geometry(load_model(run_dir), crack, run_id)
    plot_maps(npz, crack, run_id)
    plt.show()


if __name__ == "__main__":
    main()