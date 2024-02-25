#!/usr/bin/env python
# coding: utf-8
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import Affine
import numpy as np

# To create csv files
# Loop through raster bands
raster_in = "/explore/nobackup/projects/ilab/data/AGB/test/beta_pmm/MLBS_2018_541567.6_4136443.0_542567.6_4137443.0/MLBS_2018_Reflectance_reflectance_warp.tif"
raster_out = '/explore/nobackup/projects/ilab/data/AGB/test/mlruns/aggregate/MLBS_2018_Reflectance_reflectance_warp_scaled.tif'

print('reading:', str(raster_in))
out_meta = None
scaled_bands = []
with rasterio.open(raster_in) as raster1:
    out_meta = raster1.meta
    print('out_meta:', out_meta)    
    for band_no in range(1, raster1.count + 1):
        # Read band
        raster_band = raster1.read(band_no)

        scaled_raster_band = np.divide(raster_band, 10000)
        print('scaled band no:', str(band_no))
        # Append padded raster band to list
        scaled_bands.append(scaled_raster_band)
        
        # if (int(band_no) > 2):
        #     break;
print('writing:', str(raster_out))
with rasterio.open(raster_out, 'w', **out_meta) as dest:
    for band_nr, src in enumerate(scaled_bands, start=1):
        dest.write(src, band_nr)
        
print('scaling finished methusal:', str(len(scaled_bands)))
