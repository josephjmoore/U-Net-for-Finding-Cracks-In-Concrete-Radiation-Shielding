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


def build_geometry(concrete, air,
                   shield_half_width=50.0,   # shield is 100 x 100 cm
                   shield_thickness=5.0,
                   shield_z_location=30.0,   # shield centre, cm from source
                   gap_to_detector=5.0,      # shield back face -> detector layer
                   detector_thickness=1.0,   # thin scoring layer for the mesh tally
                   world_half_width=60.0,    # lateral vacuum boundary
                   world_z_min=-10.0,        # keeps the source (z=0) off the boundary
                   back_margin=10.0):        # air beyond the detector layer
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

    source_air_cell = openmc.Cell(
        name='source-side air', fill=air,
        region=world_xy & +world_min_z & -shield_min_z)

    shield_cell = openmc.Cell(
        name='shield', fill=concrete,
        region=shield_xy & shield_z)

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
 
    'scattered flux' : 0 -> 600 keV                   (1 energy bin)
    'peak flux'      : 600 keV -> line - 100 eV        (bin 0, small-angle scatter)
                       line - 100 eV -> line + 100 eV  (bin 1, uncollided)
      peak-window flux (CNN input) = bin 0 + bin 1
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
 
    peak_tally = openmc.Tally(name='peak flux')
    peak_tally.filters = [mesh_filter, openmc.EnergyFilter([
        peak_low_eV,
        line_eV - uncollided_half_window_eV,
        line_eV + uncollided_half_window_eV])]
    peak_tally.scores = ['flux']

    tallies = openmc.Tallies([scattered_tally, peak_tally])
 
    return tallies, mesh



#Caesium-137 source
source = openmc.IndependentSource()
source.space = openmc.stats.Point((0.0, 0.0, 0.0))
source.angle = openmc.stats.Isotropic()
source.energy = openmc.stats.Discrete([], [1.0]) #MeV to eV
source.particle = 'photon'
 
#settings
settings = openmc.Settings()
settings.source = source
settings.run_mode = 'fixed source'
settings.photon_transport = True
settings.batches = 10
settings.particles = 100000
 
materials = openmc.Materials([concrete])
openmc.config['cross_sections'] = cross_sections

geometry, detector_cell = build_geometry()
 


tallies = openmc.Tallies([tally, uncollided_tally, air_kerma_tally, spectrum_tally, dose_tally])
 
model = openmc.Model(geometry=geometry, materials=materials, settings=settings, tallies=tallies)

