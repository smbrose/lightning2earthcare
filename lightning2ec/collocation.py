import logging
from pathlib import Path
from typing import List, Tuple
import os 
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from satpy.modifiers.parallax import get_parallax_corrected_lonlats
from sklearn.metrics.pairwise import haversine_distances
from pyorbital.orbital import A as EARTH_RADIUS

logger = logging.getLogger(__name__)


def merge_li_datasets(directories: List[Path]) -> xr.Dataset:
    """
    Combine multiple extracted LI directories into a single xarray Dataset of BODY files.

    Args:
        directories: List of Paths to folders containing LI BODY files.

    Returns:
        A concatenated xarray.Dataset along 'groups' dimension.
    """
    body_files = []
    for folder in directories:
        if folder.is_dir():
            body_files.extend(folder.glob("*BODY*"))
    if not body_files:
        msg = "No BODY files found in provided LI directories"
        logger.error(msg)
        raise FileNotFoundError(msg)

    datasets = []
    def to_long_path(path: Path) -> str:
        p = str(path)
        if os.name == "nt" and not p.startswith("\\\\?\\"):
            p = "\\\\?\\" + os.path.abspath(p)
        return p

    for bf in body_files:
        print(f"Trying to open: {bf}")
        try:
            with xr.open_dataset(to_long_path(bf)) as ds:
                ds_mem = ds.load()
            datasets.append(ds_mem)
        except Exception as e:
            logger.warning(f"Failed to open BODY file {bf.name}: {e}")
    # Throws errors when working on windows due to to long file paths (>260 characters)         
    # for bf in body_files:
    #     print(f"Trying to open: {bf}")
    #     try:
    #         with xr.open_dataset(bf) as ds:
    #             ds_mem = ds.load()
    #         datasets.append(ds_mem)
    #     except Exception as e:
    #         logger.warning(f"Failed to open BODY file {bf.name}: {e}")

    combined = xr.concat(datasets, dim='groups', combine_attrs='drop_conflicts')
    logger.info(f"Merged {len(datasets)} BODY files into one dataset")
    return combined


def _buffer_li_indices(
    li_ds: xr.Dataset,
    shifted_lat: np.ndarray,
    shifted_lon: np.ndarray,
) -> np.ndarray:
    """
    Identify LI group indices within fixed ±0.5° spatial buffer around EarthCARE.
    """
    # Fixed buffers
    lat_buffer = 0.5
    lon_buffer = 0.5

    # Compute EarthCARE extents + buffers
    lat_min = float(np.nanmin(shifted_lat)) - lat_buffer
    lat_max = float(np.nanmax(shifted_lat)) + lat_buffer
    lon_min = float(np.nanmin(shifted_lon)) - lon_buffer
    lon_max = float(np.nanmax(shifted_lon)) + lon_buffer

    # Extract LI arrays
    li_lat  = li_ds.latitude.values
    li_lon  = li_ds.longitude.values
    li_time = li_ds.group_time.values

    # Buffer mask
    mask = (
        (li_lat >= lat_min) & (li_lat <= lat_max) &
        (li_lon >= lon_min) & (li_lon <= lon_max)
    )
    indices = np.where(mask)[0]
    logger.info(f"Buffered LI selects {len(indices)} of {li_lat.size} total groups")
    return indices


def match_li_to_ec(
    li_ds: xr.Dataset,
    cth: np.ndarray,
    ec_times: np.ndarray,
    shifted_lat: np.ndarray,
    shifted_lon: np.ndarray,
    sat_lon: float,
    sat_lat: float,
    sat_alt: float,
    time_threshold_s: int = 300,
    radius_deg: float = 0.009
) -> Tuple[xr.Dataset, np.ndarray]:
    """
    Match LI groups to EarthCARE with spatial-temporal proximity.

    Args:
        li_ds: xarray Dataset of LI groups (with coordinates latitude, longitude, group_time).
        cth: Cloud top heights (2D) from EarthCARE.
        ec_times: 1D array of EarthCARE time stamps.
        shifted_lat/lon: parallax-corrected coords, same shape as cth.
        sat_lon/lat/alt: satellite parameters for parallax computation.
        time_threshold_s: temporal threshold in seconds.
        radius_deg: spatial matching radius in degrees (~1km).

    Returns:
        matched_ds: subset of li_ds with only matched groups,
        matched_times: array of group_time stamps for matched groups.
    """
    try:
        # 1) Buffer preselection
        buf_idx = _buffer_li_indices(li_ds, shifted_lat, shifted_lon)
        if buf_idx.size == 0:
            logger.info("No LI points in buffer region; skipping match.")
            return None, None
        buffered_ds = li_ds.isel(groups=buf_idx)
        li_lat_buf = buffered_ds.latitude.values
        li_lon_buf = buffered_ds.longitude.values
        li_time_buf = buffered_ds.group_time.values

        # 2) Prepare EC
        ec_coords = np.column_stack([shifted_lat.ravel(), shifted_lon.ravel()])
        ec_coords = np.nan_to_num(ec_coords, nan=-999)
        ec_time_exp = np.repeat(ec_times, cth.shape[1])
        cth_flat    = cth.ravel()

        # 3) Spatial matching
        tree = cKDTree(ec_coords)
        pts = np.column_stack([li_lat_buf, li_lon_buf])
        dists, idxs = tree.query(pts, distance_upper_bound=radius_deg)
        spatial_mask = dists != np.inf

        # Initialize output arrays for buffered length
        n_buf = buf_idx.size
        par_lat_arr   = np.full(n_buf, np.nan)
        par_lon_arr   = np.full(n_buf, np.nan)
        time_diff_arr = np.full(n_buf, np.timedelta64('NaT'), dtype='timedelta64[ns]')

        if not np.any(spatial_mask):
            logger.info("No spatial matches within radius; skipping.")
            return None, None
        else:
            sel_buf = np.where(spatial_mask)[0]
            sel_ec  = idxs[spatial_mask]

            # 4) Temporal filtering
            li_time_sel = li_time_buf[sel_buf]
            ec_time_sel = ec_time_exp[sel_ec]
            td = li_time_sel - ec_time_sel
            time_mask = np.abs(td) <= np.timedelta64(time_threshold_s, 's')
            sel_buf_final = sel_buf[time_mask]
            sel_ec_final  = sel_ec[time_mask]

            if sel_buf_final.size == 0:
                logger.info("No temporal matches within integration window; skipping.")
                return None, None
            else:
                logger.info(f"Matched {sel_buf_final.size} LI groups to EarthCARE.")
                # Parallax correction on matched buffered points
                lon_pts = li_lon_buf[sel_buf_final]
                lat_pts = li_lat_buf[sel_buf_final]
                cth_pts = cth_flat[sel_ec_final]
                par_lon, par_lat = get_parallax_corrected_lonlats(
                    sat_lon, sat_lat, sat_alt,
                    lon_pts, lat_pts, cth_pts
                    )
                # Fill arrays
                par_lat_arr[sel_buf_final]   = par_lat
                par_lon_arr[sel_buf_final]   = par_lon
                time_diff_arr[sel_buf_final] = td[time_mask]

        # 5) Attach new variables to buffered_ds
        buffered_ds = buffered_ds.copy()
        buffered_ds['parallax_corrected_lat'] = (
            ('groups',), par_lat_arr,
            {'long_name':'Parallax corrected latitude','units':'degrees_north'}
        )
        buffered_ds['parallax_corrected_lon'] = (
            ('groups',), par_lon_arr,
            {'long_name':'Parallax corrected longitude','units':'degrees_east'}
        )
        buffered_ds['ec_time_diff'] = (
            ('groups',), time_diff_arr,
            {'long_name':'Time difference from EarthCARE'}
        )

        # Return buffered dataset and matched times
        matched_times = buffered_ds.group_time.values[np.where(~np.isnat(time_diff_arr))[0]]
        return buffered_ds, matched_times

    except Exception as e:
        logger.error(f"Error in match_li_to_ec: {e}")
        return None, None


def compute_nadir_distances(
    matched_ds: xr.Dataset,
    cpr_file: Path,
    distance_threshold_km: float = 5,
    time_threshold_s: float = 300
) -> Tuple[xr.Dataset, int]:
    """
    Compute signed cross-track distances and count close events.

    Args:
        matched_ds: xarray Dataset with parallax_corrected_lat/lon and ec_time_diff.
        cpr_file: path to CPR HDF5 for nadir coords.
        distance_threshold_km: spatial threshold for close events.
        time_threshold_s: temporal threshold in seconds.

    Returns:
        updated_ds: with new distance_from_nadir variable,
        count: number of groups within both thresholds.
    """
    try:
        cpr_ds = xr.open_dataset(cpr_file, group='ScienceData', engine='netcdf4')
        nadir_lat = cpr_ds.latitude.values
        nadir_lon = cpr_ds.longitude.values

        # Prepare parallax-corrected coordinates
        par_lat = matched_ds.parallax_corrected_lat.values
        par_lon = matched_ds.parallax_corrected_lon.values
        rad = np.radians
        coords = np.column_stack([rad(par_lat), rad(par_lon)])

        # Allocate array for signed distances
        signed_dists = np.full(par_lat.shape, np.nan)

        # Compute only for finite coordinates
        valid_mask = np.isfinite(coords).all(axis=1)
        if np.any(valid_mask):
            coords_valid = coords[valid_mask]
            nadir_coords = np.column_stack([rad(nadir_lat), rad(nadir_lon)])

            # Compute distances (radians → km)
            dists_rad = haversine_distances(coords_valid, nadir_coords)
            dists_km = dists_rad * EARTH_RADIUS

            # Determine sign based on longitude difference
            nearest_idx = np.argmin(dists_km, axis=1)
            nearest_lon = nadir_lon[nearest_idx]
            signs = np.sign(par_lon[valid_mask] - nearest_lon)
            signed_dists[valid_mask] = signs * dists_km.min(axis=1)

        # Assign distances into dataset
        updated_ds = matched_ds.copy()
        updated_ds['distance_from_nadir'] = xr.DataArray(
            signed_dists,
            dims=['groups'],
            attrs={
                'long_name': 'Signed cross-track distance to EarthCARE CPR track',
                'units': 'km',
                'description': 'Positive = east/right of track, Negative = west/left of track'
            }
        )

        # Count close
        close_mask = ~np.isnan(signed_dists) & (np.abs(signed_dists) <= distance_threshold_km)
        count = int(np.sum(close_mask))
        logger.info(f"Count within {distance_threshold_km}km & {time_threshold_s}s: {count}")

        return updated_ds, count

    except Exception as e:
        logger.error(f"Error computing nadir distances: {e}")
        raise
