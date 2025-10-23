import logging
from pathlib import Path
from typing import Tuple 
import numpy as np
import xarray as xr
import pandas as pd
from scipy.spatial import cKDTree
from satpy.modifiers.parallax import get_parallax_corrected_lonlats
from sklearn.metrics.pairwise import haversine_distances
from pyorbital.orbital import A as EARTH_RADIUS
import json

logger = logging.getLogger(__name__)


def match_li_to_ec(
    l_ds: xr.Dataset,
    cth: np.ndarray,
    ec_times: np.ndarray,
    shifted_lat: np.ndarray,
    shifted_lon: np.ndarray,
    sat_lon: float,
    sat_lat: float,
    sat_alt: float,
    time_threshold_s: int = 300,
    radius_deg: float = 0.009
) -> xr.Dataset:
    """
    Match MTG-LI lightning groups to EarthCARE with spatial-temporal proximity.

    Args:
        l_ds: xarray Dataset of lightning groups (with coordinates latitude, longitude, group_time).
        cth: Cloud top heights (2D) from EarthCARE.
        ec_times: 1D array of EarthCARE time stamps.
        shifted_lat/lon: parallax-corrected coords, same shape as cth.
        sat_lon/lat/alt: satellite parameters for parallax computation.
        time_threshold_s: temporal threshold in seconds.
        radius_deg: spatial matching radius in degrees (~1km).

    Returns:
        l_ds_new: subset of l_ds with only matched groups, plus new variables
    """
    try:
        # 1) Prepare lightning data
        l_lat = l_ds.latitude.values
        l_lon = l_ds.longitude.values
        l_time = l_ds.group_time.values

        # 2) Prepare EC
        ec_coords = np.column_stack([shifted_lat.ravel(), shifted_lon.ravel()])
        ec_coords = np.nan_to_num(ec_coords, nan=-999)
        ec_time_exp = np.repeat(ec_times, cth.shape[1])
        cth_flat    = cth.ravel()

        # 3) Spatial matching
        tree = cKDTree(ec_coords)
        pts = np.column_stack([l_lat, l_lon])
        dists, idxs = tree.query(pts, distance_upper_bound=radius_deg)
        spatial_mask = dists != np.inf

        # Initialize output arrays for buffered length
        n_buf = l_lat.size
        par_lat_arr   = np.full(n_buf, np.nan)
        par_lon_arr   = np.full(n_buf, np.nan)
        time_diff_arr = np.full(n_buf, np.timedelta64('NaT'), dtype='timedelta64[ns]')

        if not np.any(spatial_mask):
            logger.info("No spatial matches within radius; skipping.")
            return None
        else:
            sel_buf = np.where(spatial_mask)[0]
            sel_ec  = idxs[spatial_mask]

            # 4) Temporal filtering
            l_time_sel = l_time[sel_buf]
            ec_time_sel = ec_time_exp[sel_ec]
            td = l_time_sel - ec_time_sel
            time_mask = np.abs(td) <= np.timedelta64(time_threshold_s, 's')
            sel_buf_final = sel_buf[time_mask]
            sel_ec_final  = sel_ec[time_mask]

            if sel_buf_final.size == 0:
                logger.info("No temporal matches within integration window; skipping.")
                return None
            else:
                logger.info(f"Matched {sel_buf_final.size} LI groups to EarthCARE.")
                # Parallax correction on matched buffered points
                lon_pts = l_lon[sel_buf_final]
                lat_pts = l_lat[sel_buf_final]
                cth_pts = cth_flat[sel_ec_final]
                par_lon, par_lat = get_parallax_corrected_lonlats(
                    sat_lon, sat_lat, sat_alt,
                    lon_pts, lat_pts, cth_pts
                    )
                # Fill arrays
                par_lat_arr[sel_buf_final]   = par_lat
                par_lon_arr[sel_buf_final]   = par_lon
                time_diff_arr[sel_buf_final] = td[time_mask]

        # Ensure ns-resolution for writing
        group_time_ns = l_time.astype("datetime64[ns]")

        # 5) Construct a clean dataset with new variables
        l_ds_new = xr.Dataset(
            data_vars=dict(
                group_id=(
                    "groups",
                    l_ds["group_id"].values.astype("int64"),
                    {"long_name": "Group ID", "units": "1"},
                ),
                group_time=(
                    "groups",
                    group_time_ns,
                    {
                        "long_name": "Start time of integration frame",
                        "standard_name": "time",
                    },
                ),
                latitude=(
                    "groups",
                    l_ds["latitude"].values.astype("float32"),
                    {
                        "long_name": "Latitude of group",
                        "units": "degrees_north",
                        "standard_name": "latitude",
                    },
                ),
                longitude=(
                    "groups",
                    l_ds["longitude"].values.astype("float32"),
                    {
                        "long_name": "Longitude of group",
                        "units": "degrees_east",
                        "standard_name": "longitude",
                    },
                ),
                radiance=(
                    "groups",
                    l_ds["radiance"].values.astype("float32"),
                    {
                        "long_name": "Radiance of group",
                        "units": "mW.m-2.sr-1",
                    },
                ),
                flash_id=(
                    "groups",
                    l_ds["flash_id"].values.astype("int64"),
                    {"long_name": "Parent flash ID", "units": "1"},
                ),
                parallax_corrected_lat=(
                    "groups",
                    par_lat_arr.astype("float32"),
                    {
                        "long_name": "Parallax corrected latitude",
                        "units": "degrees_north",
                    },
                ),
                parallax_corrected_lon=(
                    "groups",
                    par_lon_arr.astype("float32"),
                    {
                        "long_name": "Parallax corrected longitude",
                        "units": "degrees_east",
                    },
                ),
                ec_time_diff=(
                    "groups",
                    time_diff_arr,
                    {
                        "long_name": "Time difference from EarthCARE overpass",
                    },
                ),
            )
        )
        l_ds_new["cluster_id"] = l_ds["cluster_id"].copy(deep=False)

        return l_ds_new

    except Exception as e:
        logger.error(f"Error in match_li_to_ec: {e}")
        return None


def match_glm_to_ec(
    l_ds: xr.Dataset,
    ec_times: np.ndarray,
    ec_lat: np.ndarray,
    ec_lon: np.ndarray,
    time_threshold_s: int = 300,
    radius_deg: float = 0.009
) -> xr.Dataset:
    """
    Match GOES-GLM lightning groups to EarthCARE with spatial-temporal proximity.

    Args:
        l_ds: xarray Dataset of lightning groups (with coordinates latitude, longitude, group_time).
        ec_times: 1D array of EarthCARE time stamps.
        ec_lat/lon: EarthCARE coords.
        time_threshold_s: temporal threshold in seconds.
        radius_deg: spatial matching radius in degrees (~1km).

    Returns:
        l_ds_new: subset of l_ds with only matched groups, plus new variables
    """
    try:
        # 1) Prepare lightning data
        l_lat = l_ds.latitude.values
        l_lon = l_ds.longitude.values
        l_time = l_ds.group_time.values

        # 2) Prepare EC
        ec_coords = np.column_stack([ec_lat.ravel(), ec_lon.ravel()])
        ec_coords = np.nan_to_num(ec_coords, nan=-999)
        ec_time_exp = np.repeat(ec_times, ec_lat.shape[1])

        # 3) Spatial matching
        tree = cKDTree(ec_coords)
        pts = np.column_stack([l_lat, l_lon])
        dists, idxs = tree.query(pts, distance_upper_bound=radius_deg)
        spatial_mask = dists != np.inf

        # Initialize output arrays for buffered length
        n_buf = l_lat.size
        time_diff_arr = np.full(n_buf, np.timedelta64('NaT'), dtype='timedelta64[ns]')

        if not np.any(spatial_mask):
            logger.info("No spatial matches within radius; skipping.")
            return None
        else:
            sel_buf = np.where(spatial_mask)[0]
            sel_ec  = idxs[spatial_mask]

            # 4) Temporal filtering
            l_time_sel = l_time[sel_buf]
            ec_time_sel = ec_time_exp[sel_ec]
            td = l_time_sel - ec_time_sel
            time_mask = np.abs(td) <= np.timedelta64(time_threshold_s, 's')
            sel_buf_final = sel_buf[time_mask]

            if sel_buf_final.size == 0:
                logger.info("No temporal matches within integration window; skipping.")
                return None
            else:
                logger.info(f"Matched {sel_buf_final.size} LI groups to EarthCARE.")

                # Fill arrays
                time_diff_arr[sel_buf_final] = td[time_mask]

        # Ensure ns-resolution for writing
        group_time_ns = l_time.astype("datetime64[ns]")

        # 5) Construct a clean dataset with new variables
        l_ds_new = xr.Dataset(
            data_vars=dict(
                group_id=(
                    "groups",
                    l_ds["group_id"].values.astype("int64"),
                    {"long_name": "Group ID", "units": "1"},
                ),
                group_time=(
                    "groups",
                    group_time_ns,
                    {
                        "long_name": "Start time of integration frame",
                        "standard_name": "time",
                    },
                ),
                latitude=(
                    "groups",
                    l_ds["latitude"].values.astype("float32"),
                    {
                        "long_name": "Latitude of group",
                        "units": "degrees_north",
                        "standard_name": "latitude",
                    },
                ),
                longitude=(
                    "groups",
                    l_ds["longitude"].values.astype("float32"),
                    {
                        "long_name": "Longitude of group",
                        "units": "degrees_east",
                        "standard_name": "longitude",
                    },
                ),
                radiance=(
                    "groups",
                    l_ds["radiance"].values.astype("float32"),
                    {
                        "long_name": "Radiance of group",
                        "units": "mW.m-2.sr-1",
                    },
                ),
                flash_id=(
                    "groups",
                    l_ds["flash_id"].values.astype("int64"),
                    {"long_name": "Parent flash ID", "units": "1"},
                ),
                ec_time_diff=(
                    "groups",
                    time_diff_arr,
                    {
                        "long_name": "Time difference from EarthCARE overpass",
                    },
                ),
            )
        )
        l_ds_new["cluster_id"] = l_ds["cluster_id"].copy(deep=False)

        return l_ds_new

    except Exception as e:
        logger.error(f"Error in match_li_to_ec: {e}")
        return None


# --- main: build summary; time-window first, then per-CPR helper ---
from .api_utils import fetch_earthcare_data
def build_cpr_summary2(
    matched_ds: xr.Dataset, cpr_url: str, distance_threshold_km=5.0, time_threshold_s=300
) -> tuple[xr.Dataset, int, xr.Dataset]:
    # --- load LI
    # --- load CPR dataset remotely instead of from local file
    cpr = fetch_earthcare_data(cpr_url, group="ScienceData")

    c_lat = np.asarray(cpr["latitude"].values, float)
    c_lon = np.asarray(cpr["longitude"].values, float)
    c_tim = np.asarray(pd.to_datetime(cpr["time"].values))
    c_land = np.asarray(cpr["land_flag"].values, int)

    # --- use parallax-corrected coordinates if available, otherwise fall back (for GLM cases)
    if "parallax_corrected_lat" in matched_ds and "parallax_corrected_lon" in matched_ds:
        li_lat = np.asarray(matched_ds["parallax_corrected_lat"].values, float)
        li_lon = np.asarray(matched_ds["parallax_corrected_lon"].values, float)
    else:
        li_lat = np.asarray(matched_ds["latitude"].values, float)
        li_lon = np.asarray(matched_ds["longitude"].values, float)
    li_tim = np.asarray(pd.to_datetime(matched_ds["group_time"].values))
    li_clu = np.asarray(matched_ds["cluster_id"].values)

    n_li, n_cpr = li_lat.size, c_lat.size
    # validity masks
    valid_geo_time = np.isfinite(li_lat) & np.isfinite(li_lon) & ~pd.isna(li_tim)
    # exclude noise and non-finite cluster ids globally
    not_noise = np.isfinite(li_clu) & (li_clu != -1)
    # use this everywhere
    valid_li = valid_geo_time & not_noise

    # --- nearest-in-space distance & index (for total_v1 only; unchanged)
    dmin_all = np.full(n_li, np.nan)
    nearest_cpr_idx = np.full(n_li, -1, dtype=int)
    if valid_li.any() and n_cpr:
        d_full_valid = (
            haversine_distances(
                np.column_stack([np.radians(li_lat[valid_li]), np.radians(li_lon[valid_li])]),
                np.column_stack([np.radians(c_lat), np.radians(c_lon)])
            ) * EARTH_RADIUS
        )
        nearest_idx_valid = np.argmin(d_full_valid, axis=1)
        dmin_all[valid_li] = d_full_valid[np.arange(nearest_idx_valid.size), nearest_idx_valid]
        nearest_cpr_idx[valid_li] = nearest_idx_valid

    updated_ds = matched_ds.copy()
    updated_ds["distance_from_nadir"] = xr.DataArray(
        dmin_all, dims=["groups"],
        attrs={"long_name": "Distance to closest EarthCARE CPR track point", "units": "km"}
    )

    # --- total_v1 (UNCHANGED semantics: nearest-CPR in space + time to that same CPR)
    r1 = float(distance_threshold_km)
    t1 = int(time_threshold_s)
    c_tim_s  = c_tim.astype("datetime64[s]")
    li_tim_s = li_tim.astype("datetime64[s]")

    dt_to_nearest = np.full(n_li, np.nan)
    has_nearest = nearest_cpr_idx >= 0
    if has_nearest.any():
        dt_to_nearest[has_nearest] = np.abs(
            (li_tim_s[has_nearest] - c_tim_s[nearest_cpr_idx[has_nearest]])
            .astype("timedelta64[s]").astype(np.int64)
        )

    v1_mask = valid_li & np.isfinite(dmin_all) & (dmin_all <= r1) & np.isfinite(dt_to_nearest) & (dt_to_nearest <= t1)
    total_v1 = int(np.count_nonzero(v1_mask))

    # --- per-CPR counts using "second snippet" logic (no nearest-CPR dependency)
    # Precompute radians for LI and CPR coordinates
    li_coords_rad_all = np.radians(np.column_stack((li_lat, li_lon)))
    cpr_coords_rad_all = np.radians(np.column_stack((c_lat, c_lon)))

    # Apply validity to LI once
    li_valid_idx = np.flatnonzero(valid_li)
    li_coords_rad_valid = li_coords_rad_all[li_valid_idx]
    li_tim_s_valid = li_tim_s[li_valid_idx]
    li_clu_valid = li_clu[li_valid_idx]

    # Arrays to fill
    li_count_loose  = np.zeros(n_cpr, dtype=np.int32)
    li_count_strict = np.zeros(n_cpr, dtype=np.int32)
    loose_dicts = np.empty(n_cpr, dtype=object)
    strict_dicts = np.empty(n_cpr, dtype=object)

    # thresholds
    r2 = r1 / 2.0
    t2 = t1 // 2  # strict time = half, to match your long_name and second snippet intent

    # Loop each CPR sample
    for i in range(n_cpr):
        if li_valid_idx.size == 0:
            loose_dicts[i] = {}
            strict_dicts[i] = {}
            continue

        # time masks vs this CPR sample
        dt_sec = np.abs((li_tim_s_valid - c_tim_s[i]).astype("timedelta64[s]").astype(np.int64))
        time_mask_loose = dt_sec <= t1
        time_mask_strict = dt_sec <= t2

        # If no candidates, set empty and continue
        if not time_mask_loose.any() and not time_mask_strict.any():
            loose_dicts[i] = {}
            strict_dicts[i] = {}
            continue

        cpr_coord_rad = cpr_coords_rad_all[i:i+1]  # shape (1,2)

        # LOSE mode: within r1 and t1
        if time_mask_loose.any():
            li_coords_loose = li_coords_rad_valid[time_mask_loose]
            # distances in km
            dists_km_loose = haversine_distances(cpr_coord_rad, li_coords_loose)[0] * EARTH_RADIUS
            sel_loose = dists_km_loose <= r1
            li_count_loose[i] = int(np.count_nonzero(sel_loose))

            # per-cluster dict (skip -1 and non-finite)
            if sel_loose.any():
                clusters_l = li_clu_valid[time_mask_loose][sel_loose].astype(int)
                if clusters_l.size:
                    u, c = np.unique(clusters_l, return_counts=True)
                    loose_dicts[i] = {int(k): int(v) for k, v in zip(u, c)}
                else:
                    loose_dicts[i] = {}
            else:
                loose_dicts[i] = {}
        else:
            loose_dicts[i] = {}

        # STRICT mode: within r2 and t2
        if time_mask_strict.any():
            li_coords_strict = li_coords_rad_valid[time_mask_strict]
            dists_km_strict = haversine_distances(cpr_coord_rad, li_coords_strict)[0] * EARTH_RADIUS
            sel_strict = dists_km_strict <= r2
            li_count_strict[i] = int(np.count_nonzero(sel_strict))

            # per-cluster dict (skip -1 and non-finite)
            if sel_strict.any():
                clusters_s = li_clu_valid[time_mask_strict][sel_strict].astype(int)
                if clusters_s.size:
                    u, c = np.unique(clusters_s, return_counts=True)
                    strict_dicts[i] = {int(k): int(v) for k, v in zip(u, c)}
                else:
                    strict_dicts[i] = {}
            else:
                strict_dicts[i] = {}
        else:
            strict_dicts[i] = {}

    # --- NetCDF-safe JSON for per-cluster dicts
    loose_json  = np.array([json.dumps(d, separators=(",", ":")) for d in loose_dicts])
    strict_json = np.array([json.dumps(d, separators=(",", ":")) for d in strict_dicts])

    # --- summary dataset
    cpr_summary_ds = xr.Dataset(
        data_vars=dict(
            latitude=("cpr", c_lat, {"long_name": "CPR nadir latitude", "units": "degrees_north"}),
            longitude=("cpr", c_lon, {"long_name": "CPR nadir longitude", "units": "degrees_east"}),
            time=("cpr", c_tim, {"long_name": "CPR observation time"}),
            land_flag=("cpr", c_land, {
                "long_name": "CPR land/water flag", "definition": "1 = land, 0 = not land", "units": "1"}),
            li_count_loose=("cpr", li_count_loose, {
                "long_name": f"LI groups count within ≤{r1} km and ≤{int(t1)} s", "units": "1"}),
            li_count_strict=("cpr", li_count_strict, {
                "long_name": f"LI groups count within ≤{r2} km and ≤{int(t2)} s", "units": "1"}),
            li_count_loose_per_cluster=("cpr", loose_json, {
                "long_name": "Loose per-cluster counts (JSON: {cluster_id: count})", "content_encoding": "json"}),
            li_count_strict_per_cluster=("cpr", strict_json, {
                "long_name": "Strict per-cluster counts (JSON: {cluster_id: count})", "content_encoding": "json"}),
        )
    )

    return updated_ds, total_v1, cpr_summary_ds
