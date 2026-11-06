import openmc
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

cross_sections = os.environ.get(
    'OPENMC_CROSS_SECTIONS', '/home/asus/nuclear_data/endfb-viii.0-hdf5/cross_sections.xml')

#Make concrete material
concrete = openmc.Material(name='concrete')
for element, fraction in [('H', 0.0221), ('C', 0.0025), ('O', 0.5749),
                            ('Na', 0.0152), ('Mg', 0.0013), ('Al', 0.0200),
                            ('Si', 0.3046), ('K', 0.0100), ('Ca', 0.0430),
                            ('Fe', 0.0064)]:
    concrete.add_element(element, fraction, percent_type='wo')
concrete.set_density('g/cm3', 2.3)

air = openmc.Material(name='air')
for element, fraction in [('C', 0.000124), ('N', 0.755268),
                            ('O', 0.231781), ('Ar', 0.012827)]:
    air.add_element(element, fraction, percent_type='wo')
air.set_density('g/cm3', 0.001205)

def sample_crack(rng, scan_half_width=20.0, shield_thickness=5.0):
    length = rng.uniform(2.0, 20.0)
    width = 10 ** rng.uniform(np.log10(0.05), np.log10(1.0))   # log-uniform, 0.5 mm – 1 cm
    depth = rng.uniform(0.5, shield_thickness)
    phi = rng.uniform(0.0, np.pi)

    margin = scan_half_width - length / 2      # keeps the whole crack inside the scan area
    x = rng.uniform(-margin, margin)
    y = rng.uniform(-margin, margin)

    return {"x": x, "y": y, "length": length, "width": width,
            "depth": depth, "phi": phi}


def build_geometry(concrete, air,
                   shield_half_width=50.0,   # shield is 100 x 100 cm
                   shield_thickness=5.0,
                   shield_z_location=30.0,   # shield centre, cm from source
                   gap_to_detector=5.0,      # shield back face -> detector layer
                   detector_thickness=1.0,   # thin scoring layer for the mesh tally
                   world_half_width=60.0,    # lateral vacuum boundary
                   world_z_min=-10.0,        # keeps the source (z=0) off the boundary
                   back_margin=10.0,         # air beyond the detector layer
                   crack = None              # default no crack
                    ):           
    """
    Point source at the origin, concrete slab along +z, air everywhere else:
        -10.0 -> 27.5  air (source side)
         27.5 -> 32.5  concrete shield (air around its lateral edges)
         32.5 -> 37.5  air gap
         37.5 -> 38.5  air detector layer (put the RegularMesh here)
         38.5 -> 48.5  air (back region)
    All outer surfaces are vacuum boundaries.

    Returns (geometry, detector_cell).
    """
    shield_front = shield_z_location - shield_thickness / 2
    shield_back = shield_z_location + shield_thickness / 2
    detector_front = shield_back + gap_to_detector
    detector_back = detector_front + detector_thickness
    world_z_max = detector_back + back_margin

    # Outer boundary: the only vacuum surfaces
    world_min_x = openmc.XPlane(x0=-world_half_width, boundary_type='vacuum')
    world_max_x = openmc.XPlane(x0=world_half_width, boundary_type='vacuum')
    world_min_y = openmc.YPlane(y0=-world_half_width, boundary_type='vacuum')
    world_max_y = openmc.YPlane(y0=world_half_width, boundary_type='vacuum')
    world_min_z = openmc.ZPlane(z0=world_z_min, boundary_type='vacuum')
    world_max_z = openmc.ZPlane(z0=world_z_max, boundary_type='vacuum')

    # Shield
    shield_min_x = openmc.XPlane(x0=-shield_half_width)
    shield_max_x = openmc.XPlane(x0=shield_half_width)
    shield_min_y = openmc.YPlane(y0=-shield_half_width)
    shield_max_y = openmc.YPlane(y0=shield_half_width)
    shield_min_z = openmc.ZPlane(z0=shield_front)
    shield_max_z = openmc.ZPlane(z0=shield_back)

    # Detector layer
    detector_min_z = openmc.ZPlane(z0=detector_front)
    detector_max_z = openmc.ZPlane(z0=detector_back)

    world_xy = +world_min_x & -world_max_x & +world_min_y & -world_max_y
    shield_xy = +shield_min_x & -shield_max_x & +shield_min_y & -shield_max_y
    shield_z = +shield_min_z & -shield_max_z

    #Crack
    crack_region = None
    if crack is not None:
        x0, y0 = crack["x"], crack["y"]
        l, w , d = crack["length"], crack["width"], crack["depth"]
        c, s = np.cos(crack["phi"]), np.sin(crack["phi"])

        u_c = x0 * c + y0 * s       # centre along the crack
        n_c = -x0 * s + y0 * c      # centre across the crack

        end_lo  = openmc.Plane(a=c,  b=s, c=0.0, d=u_c - l / 2)
        end_hi  = openmc.Plane(a=c,  b=s, c=0.0, d=u_c + l / 2)
        wall_lo = openmc.Plane(a=-s, b=c, c=0.0, d=n_c - w / 2)
        wall_hi = openmc.Plane(a=-s, b=c, c=0.0, d=n_c + w / 2)

        if d >= shield_thickness:
            crack_bottom = shield_min_z        
        else:
            crack_bottom = openmc.ZPlane(z0=shield_back - d)

        crack_region = (+end_lo & -end_hi & +wall_lo & -wall_hi
                        & +crack_bottom & -shield_max_z
                        & shield_xy)

    source_air_cell = openmc.Cell(
        name='source-side air', fill=air,
        region=world_xy & +world_min_z & -shield_min_z)

    shield_region = shield_xy & shield_z
    if crack_region is not None:
        shield_region = shield_region & ~crack_region

    shield_cell = openmc.Cell(name='shield', fill=concrete, region=shield_region)


    shield_surround_cell = openmc.Cell(
        name='air around shield edges', fill=air,
        region=world_xy & shield_z & ~shield_xy)

    gap_cell = openmc.Cell(
        name='gap air', fill=air,
        region=world_xy & +shield_max_z & -detector_min_z)

    detector_cell = openmc.Cell(
        name='detector layer', fill=air,
        region=world_xy & +detector_min_z & -detector_max_z)

    back_air_cell = openmc.Cell(
        name='back air', fill=air,
        region=world_xy & +detector_max_z & -world_max_z)

    cells = [source_air_cell, shield_cell, shield_surround_cell,
             gap_cell, detector_cell, back_air_cell]

    if crack_region is not None:
        cells.append(openmc.Cell(name='crack', fill=air, region=crack_region))


    root = openmc.Universe(cells=cells)
    geometry = openmc.Geometry(root)
    return geometry, detector_cell

def build_tallies(detector_cell,
                  n_pixels=64,
                  scan_half_width=20.0,      # 40 x 40 cm scan area
                  peak_low_eV=600e3,         
                  line_eV=662e3,
                  uncollided_half_window_eV=100.0):
    """
    Two mesh tallies over the scan area, inside the air detector layer.
 
        'scattered flux' 0 -> 600 keV (bin 0)
        'peak flux' 600 keV -> line - 100 eV (bin 0, small-angle scatter)
                    line - 100 eV -> line + 100 eV  (bin 1, uncollided)

      peak-window flux (CNN input) = bin 0 + bin 1 (shape 64*64*2)
      uncollided flux (check against S*exp(-mu*t)/(4*pi*r^2)) = bin 1
    """
    bb = detector_cell.bounding_box # gets bounds
    z_min , z_max = bb[0][2], bb[1][2] 
 
    mesh = openmc.RegularMesh(name='scan mesh')
    mesh.dimension = [n_pixels, n_pixels, 1]
    mesh.lower_left = [-scan_half_width, -scan_half_width, z_min]
    mesh.upper_right = [scan_half_width, scan_half_width, z_max]
    mesh_filter = openmc.MeshFilter(mesh)
 
    scattered_tally = openmc.Tally(name='scattered flux')
    scattered_tally.filters = [mesh_filter, openmc.EnergyFilter([0.0, peak_low_eV])]
    scattered_tally.scores = ['flux']
 
    # Photons scattered at small angles: 600-661.9, uncollided photons: 661.9 to 662.1 keV
    peak_tally = openmc.Tally(name='peak flux')
    peak_tally.filters = [mesh_filter, openmc.EnergyFilter([
        peak_low_eV,
        line_eV - uncollided_half_window_eV,
        line_eV + uncollided_half_window_eV])]
    peak_tally.scores = ['flux']

    tallies = openmc.Tallies([scattered_tally, peak_tally])
 
    return tallies, mesh

def flux_map(sp_path, mesh, source_rate):
    nx, ny, nz = mesh.dimension
    voxel_volume = np.prod((np.array(mesh.upper_right) - np.array(mesh.lower_left))
                           / np.array(mesh.dimension))

    with openmc.StatePoint(sp_path) as sp:
        peak = sp.get_tally(name='peak flux')
        scat = sp.get_tally(name='scattered flux')

        # Bins are flattened as (mesh bin, energy bin), with energy changing fastest
        # and mesh x changing fastest within the mesh -> reshape to (z, y, x, E)
        peak_mean = peak.mean.reshape(nz, ny, nx, 2)[0]
        scat_mean = scat.mean.reshape(nz, ny, nx, 1)[0]
        peak_std  = peak.std_dev.reshape(nz, ny, nx, 2)[0]
        scat_std  = scat.std_dev.reshape(nz, ny, nx, 1)[0]

    # Raw tally cm per source particle -> divide by volume
    # to get cm^-2, multiply by photons/s to get cm^-2s^-1
    norm = source_rate / voxel_volume
    return {
        "peak":       (peak_mean[..., 0] + peak_mean[..., 1]) * norm,  # CNN input
        "uncollided": peak_mean[..., 1] * norm,                     
        "scattered":  scat_mean[..., 0] * norm,
        "peak_std":   np.sqrt(peak_std[..., 0]**2 + peak_std[..., 1]**2) * norm,
        "scattered_std": scat_std[..., 0] * norm,
    }


crack_parameter_label = ("x", "y", "width", "length", "depth", "sin2phi", "cos2phi")


def crack_to_params(crack):
    """Fixed-length float32 array of crack parameters; all NaN if there is no crack."""
    if crack is None:
        return np.full(len(crack_parameter_label), np.nan, dtype=np.float32)
    values = dict(crack,
                  sin2phi=np.sin(2 * crack["phi"]),
                  cos2phi=np.cos(2 * crack["phi"]))
    return np.array([values[label] for label in crack_parameter_label], dtype=np.float32)



def save_run(path, maps, crack):
    """Save one simulation as a compressed .npz."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image=np.stack([maps["peak"], maps["scattered"]]).astype(np.float32),  # (2, ny, nx)
        image_std=np.stack([maps["peak_std"], maps["scattered_std"]]).astype(np.float32),
        uncollided=maps["uncollided"].astype(np.float32),
        has_crack=np.bool_(crack is not None),
        params=crack_to_params(crack),
        param_labels=np.array(crack_parameter_label),
    )


# Caesium-137 source
cs137_photon = 661.657e3   
factor = 0.851   # 662 keV gammas per decay
activity = 1.0e9        

source = openmc.IndependentSource()
source.space = openmc.stats.Point((0.0, 0.0, 0.0))
source.angle = openmc.stats.Isotropic()
source.energy = openmc.stats.Discrete([cs137_photon], [1.0])  # eV
source.particle = 'photon'

# settings
settings = openmc.Settings()
settings.source = source
settings.run_mode = 'fixed source'
settings.photon_transport = True
settings.batches = 10
settings.particles = 4000000

materials = openmc.Materials([concrete, air])
openmc.config['cross_sections'] = cross_sections


def run_one(i, crack=None, runs_dir="runs", data_dir="data"):
    """Build, run and save simulation i. crack: None or dict keyed by CRACK_PARAM_NAMES."""
    geometry, detector_cell = build_geometry(concrete, air, crack=crack)
    tallies, mesh = build_tallies(detector_cell, line_eV=cs137_photon)

    settings.seed = 1234 + i
    model = openmc.Model(geometry=geometry, materials=materials,
                         settings=settings, tallies=tallies)
    
    # a directory per run so statepoints don't overwrite    
    sp_path = model.run(cwd=f"{runs_dir}/run_{i:05d}")

    maps = flux_map(sp_path, mesh, source_rate=activity * factor)
    out_path = f"{data_dir}/run_{i:05d}.npz"
    save_run(out_path, maps, crack)
    return out_path


def generate_dataset(n_runs, start=0):
    for i in range(start, start + n_runs):
        rng = np.random.default_rng()
        if rng.uniform() < 0.5:
            crack = sample_crack(rng)
        else:
            crack = None
        print(f"run {i}: saved {run_one(i, crack)}")


if __name__ == "__main__":
    generate_dataset(n_runs=1)
