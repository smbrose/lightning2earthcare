import logging
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)


def interpolate_cloud_top_heights(cth: np.ndarray) -> np.ndarray:
    """
    Fill missing cloud top heights (NaNs) using nearest-neighbor interpolation.

    Args:
        cth: 2D array of cloud top heights (meters), may contain NaNs.

    Returns:
        A 2D array with NaNs replaced by nearest valid neighbor values.
    """
    # Create grid indices
    rows, cols = np.indices(cth.shape)
    points = np.column_stack((rows.ravel(), cols.ravel()))
    values = cth.ravel()

    # Mask of valid points
    valid_mask = ~np.isnan(values)
    valid_points = points[valid_mask]
    valid_values = values[valid_mask]

    # Interpolate onto full grid
    grid_rows, grid_cols = np.mgrid[0:cth.shape[0], 0:cth.shape[1]]
    filled = griddata(valid_points, valid_values, (grid_rows, grid_cols), method='nearest')

    return filled


def prepare_ec(cop_file: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess EarthCARE COP product to extract coordinates and cloud-top heights.

    Args:
        cop_file: Path to the COP HDF5 file (group 'ScienceData').

    Returns:
        lon: 2D array of longitudes (degrees east).
        lat: 2D array of latitudes (degrees north).
        cth_filled: 2D array of cloud-top heights (meters), NaNs filled.
        times: 1D array of time stamps (numpy.datetime64).
    """
    try:
        ds = xr.open_dataset(cop_file, group='ScienceData', engine='netcdf4')
    except Exception as e:
        logger.error(f"Failed to open EarthCARE COP file {cop_file}: {e}")
        raise

    # Extract variables
    lon = ds.longitude.values
    lat = ds.latitude.values
    cth = ds.cloud_top_height.values
    times = ds.time.values

    # Mask out any columns with invalid coordinates
    valid_mask = ~np.isnan(lat).any(axis=0)
    lon_clipped = lon[:, valid_mask]
    lat_clipped = lat[:, valid_mask]
    cth_clipped = cth[:, valid_mask]

    logger.debug(
        f"Clipped to {lon_clipped.shape[1]} valid swath columns (from {lon.shape[1]})"
    )

    # Fill missing cloud top heights
    cth_filled = interpolate_cloud_top_heights(cth_clipped)

    return lon_clipped, lat_clipped, cth_filled, times
