import logging
from pathlib import Path

import numpy as np
import xarray as xr
import glob
from scipy.interpolate import griddata
from shapely.geometry import Polygon
from shapely import vectorized
import os
from typing import List

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
        with xr.open_dataset(cop_file, group='ScienceData', engine='netcdf4') as ds:
            # Extract variables
            lon = ds.longitude.values
            lat = ds.latitude.values
            cth = ds.cloud_top_height.values
            times = ds.time.values
    except Exception as e:
        logger.error(f"Failed to open EarthCARE COP file {cop_file}: {e}")
        raise

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

from .api_utils import fetch_earthcare_data
def prepare_ec2(msi_url):
    """
    Prepare EarthCARE MSI data for processing.

    Opens the MSI dataset (locally or via HTTPS STAC asset),
    extracts key fields, and returns them as numpy arrays.
    """
    logger.info(f"Fetching MSI dataset from {msi_url}")

    msi_ds = fetch_earthcare_data(msi_url)

    required_vars = ["longitude", "latitude", "cloud_top_height", "time"]
    for var in required_vars:
        if var not in msi_ds:
            raise KeyError(f"Missing expected variable '{var}' in MSI dataset")

    lon = msi_ds["longitude"].values
    lat = msi_ds["latitude"].values
    cth = msi_ds["cloud_top_height"].values
    ec_times = msi_ds["time"].values

    logger.info(
        f"Loaded MSI dataset: lon({lon.shape}), lat({lat.shape}), "
        f"cth({cth.shape}), time({len(ec_times)})"
    )
    return lon, lat, cth, ec_times


def merge_li_datasets(nc_files: List[Path]) -> xr.Dataset:
    """
    Combine multiple LI netcdf "BODY" files into a single xarray Dataset.

    Args:
        nc_files: List of Paths to LI BODY files.

    Returns:
        A concatenated xarray.Dataset along 'groups' dimension.
    """

    datasets = []

    for bf in nc_files:
        logger.info(f"Trying to open BODY file: {bf}")
        try:
            with xr.open_dataset(bf, engine='h5netcdf') as ds:
                ds_mem = ds.load()  # Load into memory to avoid lazy loading issues
            datasets.append(ds_mem)
        except Exception as e:
            logger.warning(f"Failed to open BODY file {bf}: {e}")

    if not datasets:
        logger.error("No BODY files could be opened successfully.")
        return None

    # Concatenate all datasets along the 'groups' dimension
    try:
        combined_ds = xr.concat(datasets, dim="groups")
        logger.info(f"Successfully merged {len(datasets)} BODY files into a single dataset.")
        return combined_ds
    except Exception as e:
        logger.error(f"Failed to concatenate BODY files: {e}")
        return None


def buffer_li(
    li_ds: xr.Dataset,
    shifted_lat: np.ndarray,
    shifted_lon: np.ndarray,
    buffer_deg: float = 0.5,
) -> np.ndarray:
    """
    Identify LI group indices within a 0.5° buffer of the EarthCARE swath shape.

    Returns indices of li_ds groups whose lat/lon fall inside buffered region.
    """
    nrows, ncols = shifted_lat.shape

    # Finite in both lat & lon
    finite = np.isfinite(shifted_lat) & np.isfinite(shifted_lon)

    left_edge, right_edge = [], []
    for i in range(nrows):
        cols = np.flatnonzero(finite[i])
        if cols.size == 0:
            continue  # this row has only NaNs -> skip
        jL, jR = cols[0], cols[-1]
        left_edge.append((shifted_lon[i, jL], shifted_lat[i, jL]))
        right_edge.append((shifted_lon[i, jR], shifted_lat[i, jR]))

    ring = left_edge + right_edge[::-1]

    outline = Polygon(ring)
    region = outline.buffer(buffer_deg)

    li_lat = li_ds.latitude.values
    li_lon = li_ds.longitude.values

    minx, miny, maxx, maxy = region.bounds
    in_bbox = (li_lon >= minx) & (li_lon <= maxx) & (li_lat >= miny) & (li_lat <= maxy)
    if not np.any(in_bbox):
        return None

    # Only points within bounding box:
    li_lon_bbox = li_lon[in_bbox]
    li_lat_bbox = li_lat[in_bbox]
    indices_bbox = np.where(in_bbox)[0]

    mask_in_poly = vectorized.contains(region, li_lon_bbox, li_lat_bbox)
    indices = indices_bbox[mask_in_poly]

    if indices.size == 0:
        return None
    else:
        logger.info(f"Buffer LI selects {len(indices)} of {li_lat.size} total groups")
        buffered_ds = li_ds.isel(groups=indices)
        return buffered_ds
