import logging
from pathlib import Path
from typing import List, Tuple

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
    for bf in body_files:
        try:
            ds = xr.open_dataset(bf)
            datasets.append(ds)
        except Exception as e:
            logger.warning(f"Failed to open BODY file {bf.name}: {e}")

    combined = xr.concat(datasets, dim='groups', combine_attrs='drop_conflicts')
    logger.info(f"Merged {len(datasets)} BODY files into one dataset")
    return combined


def match_li_to_ec(
    li_ds: xr.Dataset,
    cth: np.ndarray,
    ec_times: np.ndarray,
    shifted_lat: np.ndarray,
    shifted_lon: np.ndarray,
    sat_lon: float,
    sat_lat: float,
    sat_alt: float,
    li_integration: int,
    radius_deg: float = 0.009
) -> Tuple[xr.Dataset, np.ndarray]:
    """
    Spatial-temporal match of LI observations to EarthCARE swath points.

    Args:
        li_ds: xarray Dataset of LI groups (with coordinates latitude, longitude, group_time).
        cth: Cloud top heights (2D) from EarthCARE.
        ec_times: 1D array of EarthCARE time stamps.
        shifted_lat/lon: parallax-corrected coords, same shape as cth.
        sat_lon/lat/alt: satellite parameters for parallax computation.
        li_integration: max time difference.
        radius_deg: spatial matching radius in degrees (~1km).

    Returns:
        matched_ds: subset of li_ds with only matched groups,
        matched_times: array of group_time stamps for matched groups.
    """
    try:
        # Extract LI coordinates and times
        li_lat = li_ds.latitude.values
        li_lon = li_ds.longitude.values
        li_time = li_ds.group_time.values
        li_group = li_ds.group_id.values

        # Prepare EarthCARE spatial-temporal arrays
        ec_coords = np.column_stack([shifted_lat.ravel(), shifted_lon.ravel()])
        ec_coords = np.nan_to_num(ec_coords, nan=-999)
        ec_time_expanded = np.repeat(ec_times, cth.shape[1])
        cth_flat = cth.ravel()

        # Spatial matching via KDTree
        tree = cKDTree(ec_coords)
        query_pts = np.column_stack([li_lat, li_lon])
        distances, indices = tree.query(query_pts, distance_upper_bound=radius_deg)
        spatial_mask = distances != np.inf
        if not np.any(spatial_mask):
            logger.warning("No valid spatial matches found within radius")
            return None, None

        # Filter by valid spatial matches
        spatial_idx = indices[spatial_mask]
        spatial_groups = li_group[spatial_mask]

        # Temporal matching
        matched_ec_times = ec_time_expanded[spatial_idx]
        matched_cth = cth_flat[spatial_idx]
        time_diff = li_time[spatial_mask] - matched_ec_times
        time_mask = np.abs(time_diff) <= np.timedelta64(li_integration, 'm')
        if not np.any(time_mask):
            logger.warning("No temporal matches within integration window")
            return None, None

        # Final matched IDs and times
        matched_ids = spatial_groups[time_mask]
        matched_times = li_time[spatial_mask][time_mask]
        logger.info(f"Number of matched lightning observations: {len(matched_times)}")

        # Subset LI dataset
        li_matched = li_ds.where(li_ds.group_id.isin(matched_ids), drop=True)

        # Parallax-correct LI points
        li_lon_pts = li_lon[spatial_mask][time_mask]
        li_lat_pts = li_lat[spatial_mask][time_mask]
        li_cth_pts = matched_cth[time_mask]
        par_lon, par_lat = get_parallax_corrected_lonlats(
            sat_lon, sat_lat, sat_alt,
            li_lon_pts, li_lat_pts, li_cth_pts
        )

        # Build additional variables
        new_vars = xr.Dataset(
            {
                'parallax_corrected_lat': xr.DataArray(
                    par_lat, dims=['groups'],
                    attrs={
                        'long_name': 'Parallax corrected latitude',
                        'units': 'degrees_north'
                    }
                ),
                'parallax_corrected_lon': xr.DataArray(
                    par_lon, dims=['groups'],
                    attrs={
                        'long_name': 'Parallax corrected longitude',
                        'units': 'degrees_east'
                    }
                ),
                'ec_time_diff': xr.DataArray(
                    time_diff[time_mask], dims=['groups'],
                    attrs={
                        'long_name': 'Time difference from EarthCARE acquisition'
                    }
                )
            },
            coords={**li_matched.coords}
        )

        # Merge and return
        matched_ds = xr.merge([li_matched, new_vars])
        logger.info("Matched dataset created with parallax corrections")
        return matched_ds, matched_times

    except Exception as e:
        logger.error(f"Error in spatial-temporal matching: {e}")
        return None, None


def compute_nadir_distances(
    matched_ds: xr.Dataset,
    cpr_file: Path,
    distance_threshold_km: float = 2.5,
    time_threshold_s: float = 150
) -> Tuple[xr.Dataset, int]:
    """
    Compute signed cross-track distances and count close events.

    Args:
        matched_ds: xarray Dataset with parallax_corrected_lat/lon and ec_time_diff.
        cpr_file: path to CPR HDF5 for nadir coords.
        distance_threshold_km: spatial threshold for close events.
        time_threshold_s: temporal threshold in seconds.

    Returns:
        matched_ds: with new distance_from_nadir variable,
        count: number of groups within both thresholds.
    """
    try:
        cpr_ds = xr.open_dataset(cpr_file, group='ScienceData', engine='netcdf4')
        nadir_lat = cpr_ds.latitude.values
        nadir_lon = cpr_ds.longitude.values

        # compute haversine distances
        rad = np.radians
        li_coords = np.column_stack([
            rad(matched_ds.parallax_corrected_lat.values),
            rad(matched_ds.parallax_corrected_lon.values)
        ])
        nadir_coords = np.column_stack([rad(nadir_lat), rad(nadir_lon)])
        dists_rad = haversine_distances(li_coords, nadir_coords)
        dists_km = dists_rad * EARTH_RADIUS

        # signed by lon difference
        idx_min = np.argmin(dists_km, axis=1)
        nearest_nadir_lon = nadir_lon[idx_min]
        signs = np.sign(
            matched_ds.parallax_corrected_lon.values - nearest_nadir_lon
        )
        signed_km = signs * dists_km.min(axis=1)

        matched_ds['distance_from_nadir'] = (
            ('groups',), signed_km,
            {
                'long_name': 'Signed cross-track distance to EarthCARE CPR track',
                'units': 'km',
                'description': 'Positive = east/right of track, Negative = west/left of track'
            }
        )

        # count close
        time_diff_s = np.abs(
            matched_ds.ec_time_diff.values.astype('timedelta64[s]').astype(int)
        )
        mask_close = (np.abs(signed_km) <= distance_threshold_km) & (time_diff_s <= time_threshold_s)
        count = int(mask_close.sum())
        logger.info(f"Count within {distance_threshold_km}km & {time_threshold_s}s: {count}")
        return matched_ds, count

    except Exception as e:
        logger.error(f"Error computing nadir distances: {e}")
        raise
