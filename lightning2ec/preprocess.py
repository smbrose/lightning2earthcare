import logging
from pathlib import Path

import numpy as np
import xarray as xr
import glob
from scipy.interpolate import griddata
from shapely.geometry import Polygon
from shapely import vectorized

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


def merge_li_datasets(li_paths: list[Path]) -> xr.Dataset | None:
    """
    Combine multiple extracted LI directories into a single xarray Dataset of BODY files.

    Args:
        li_paths: List of Paths to folders containing LI BODY files.

    Returns:
        A concatenated xarray.Dataset along 'groups' dimension.
    """
    datasets = []
    n_found = 0
    n_ok = 0
    for p in li_paths:
        for nc in sorted(glob.glob(str(p / "*BODY*.nc"))):
            n_found += 1
            try:
                ds = xr.open_dataset(nc, engine="netcdf4")
                datasets.append(ds)
                n_ok += 1
            except Exception as e:
                logger.warning(f"Failed to open BODY file {nc}: {e}")

    if not datasets:
        logger.info(f"merge_li_datasets: 0/{n_found} BODY files opened successfully; returning None.")
        return None

    logger.info(f"merge_li_datasets: opened {n_ok}/{n_found} BODY files; concatenating.")
    try:
        combined = xr.concat(datasets, dim="groups", combine_attrs="drop_conflicts")
    finally:
        for ds in datasets:
            ds.close()
    return combined


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
