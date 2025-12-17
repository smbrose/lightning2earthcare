import numpy as np
import logging
import gc
from .preprocess import merge_li_datasets, buffer_lightning_data
from .clustering import cluster_lightning_groups, subcluster_lightning_groups
from .parallax import apply_parallax_shift
from .collocation import match_li_to_ec, match_glm_to_ec, build_cpr_summary2
from .netcdf_writer import write_lightning_netcdf, write_track_netcdf

logger = logging.getLogger(__name__)


SAT_GEOM = {
    "MTG-I1":  {"lon": 0.0,     "lat": 0.0, "alt": 35786400.0},
    "GOES-16": {"lon": -75.2,   "lat": 0.0, "alt": 35786400.0},  # East
    "GOES-18": {"lon": -137.0,  "lat": 0.0, "alt": 35786400.0},  # West
    "GOES-19": {"lon": -75.2,   "lat": 0.0, "alt": 35786400.0},  # East
}


def process_one_source(
    source_key,
    platform,
    t0, t1,
    l_base,
    orbit_frame,
    ec_lon, ec_lat, ec_cth, ec_times,
    cpr_url,
    distance_threshold_km,
    time_threshold_s
):
    """Run the full pipeline for a SINGLE lightning source."""

    # 1) Download/open + merge
    if source_key == 'LI':
        from .download import download_li
        paths = download_li(t0, t1, l_base)
        if not paths:
            logger.info(f"[{source_key}] no input files downloaded; skipping.")
            return
        ds_merged = merge_li_datasets(paths)
    else:
        from .download import load_merge_glm
        ds_merged = load_merge_glm(t0, t1, platform)

    if ds_merged is None:
        logger.info(f"[{source_key}] no usable files after merge; skipping.")
        return

    # 2) Buffer
    buf_ds = buffer_lightning_data(ds_merged, ec_lat, ec_lon)
    if buf_ds is None:
        logger.info(f"[{source_key}] no points in buffer; skipping.")
        return
    del ds_merged
    gc.collect()

    # 3) Cluster
    clustered = cluster_lightning_groups(buf_ds, eps=5.0, time_weight=0.5, min_samples=20, lat_gap=0.25)
    if clustered is None:
        logger.info(f"[{source_key}] no clusters; skipping.")
        return
    del buf_ds
    gc.collect()
    
    if source_key == 'LI':
        # 4) Parallax shift
        geom = SAT_GEOM[platform]
        sat_lon, sat_lat, sat_alt = geom["lon"], geom["lat"], geom["alt"]

        shifted_lat, shifted_lon = apply_parallax_shift(
            ec_lon, ec_lat, ec_cth,
            sat_lon, sat_lat, sat_alt
        ) 

        # 5) Collocate
        matched_ds = match_li_to_ec(
            clustered, ec_cth, ec_times,
            shifted_lat, shifted_lon,
            sat_lon, sat_lat, sat_alt,
            time_threshold_s=time_threshold_s
        )
    else:
        # 4-5) Collocate directly (no parallax for GLM)
        # GLM L2 data already include lite parallax correction (implement lower ellipsoid)
        matched_ds = match_glm_to_ec(
            clustered, ec_times,
            ec_lat, ec_lon,
            time_threshold_s=time_threshold_s
        )
    del clustered
    gc.collect()

    if matched_ds is None:
        logger.info(f"[{source_key}] no matches found; skipping.")
        return
    
    # NEW: Sub-cluster matched points
    subclustered = subcluster_lightning_groups(matched_ds, eps=5.0, time_weight=0.5, min_samples=20)
    del matched_ds

    # 6) CPR summary
    final_ds, close_count, track_ds = build_cpr_summary2(
        subclustered, cpr_url,
        distance_threshold_km=distance_threshold_km,
        time_threshold_s=time_threshold_s
    )
    del subclustered
    gc.collect()
    # skip this orbit if CPR couldn’t be loaded
    if final_ds is None:
        logger.error(f"[{orbit_frame}/{source_key}] CPR summary could not be built; skipping this orbit/source.")
        return
    
    # 7) Write
    write_lightning_netcdf(final_ds, l_base, orbit_frame, close_count, source_label=source_key, platform=platform)
    if np.max(track_ds.li_count_loose.values) > 0:
        write_track_netcdf(track_ds, l_base, orbit_frame, close_count, source_label=source_key, platform=platform)
