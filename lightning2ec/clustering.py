import logging
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import DBSCAN
from pyproj import Transformer

logger = logging.getLogger(__name__)


def filter_clusters_by_quality(
    ds: xr.Dataset,
    cluster_bad_threshold: float = 0.25
) -> xr.Dataset:
    """
    Filters out low-quality clusters and groups based on flag variables.
    Works for MTG-LI and GLM:
      - GLM: uses 'group_quality_flag' (0 = good)
      - LI : uses the existing l1b_* warnings (0 = good)
    Drops flag and auxiliary variables after filtering.
    """
    cluster_id_var = "cluster_id"

    # Candidate flags across products (0 = good)
    candidate_flags = [
        "group_quality_flag",          # GLM
        "l1b_missing_warning",         # LI
        "l1b_geolocation_warning",     # LI
        "l1b_radiometric_warning",     # LI
    ]
    # Only those actually present in the dataset
    flag_vars = [f for f in candidate_flags if f in ds.variables]

    auxiliary_vars = [
        "auxiliary_dataset_identifier",
        "auxiliary_dataset_status",
        "group_filter_qa",
    ]

    # If no known flags are present, return unchanged
    if not flag_vars:
        logger.info("No known quality flags found; skipping quality filtering.")
        return ds

    # Per-group quality mask: keep only flag == 0 for all present flags
    n_groups = ds.sizes.get("groups", None)
    if n_groups is None:
        logger.warning("Dataset missing 'groups' dimension; skipping quality filtering.")
        return ds

    quality_mask = np.ones(n_groups, dtype=bool)
    for flag in flag_vars:
        arr = ds[flag]
        # We only screen group-level flags; non-group dims or scalars are ignored
        if "groups" in getattr(arr, "dims", ()):
            quality_mask &= (arr.values == 0)

    # Build per-cluster bad ratio (ignore noise cluster -1)
    cluster_ids = ds[cluster_id_var].values
    df = ds[[cluster_id_var]].to_dataframe().reset_index()
    df["is_bad"] = ~quality_mask
    df = df[df[cluster_id_var] != -1]

    if not df.empty:
        bad_rates = df.groupby(cluster_id_var)["is_bad"].mean()
        bad_clusters = bad_rates[bad_rates > cluster_bad_threshold].index.values
        is_in_bad_cluster = np.isin(cluster_ids, bad_clusters)
        final_mask = (~is_in_bad_cluster) & quality_mask
        logger.info(f"Dropped {len(bad_clusters)} of {bad_rates.size} clusters for quality issues.")
    else:
        # No clusters (or all noise) — just use per-group quality
        final_mask = quality_mask
        logger.info("No non-noise clusters; applying per-group quality mask only.")

    # Subset dataset
    ds = ds.isel(groups=final_mask)

    # Drop flag and auxiliary variables (include any flags we used)
    vars_to_drop = [v for v in (flag_vars + auxiliary_vars) if v in ds.variables]
    if vars_to_drop:
        ds = ds.drop_vars(vars_to_drop)

    return ds


def cluster_lightning_groups(
    li_ds: xr.Dataset,
    eps: float = 5.0,
    time_weight: float = 0.5,
    min_samples: int = 20,
    lat_gap: float = 0.25
) -> xr.Dataset:
    """
    Cluster LI groups in space-time using DBSCAN, after splitting into latitude chunks.
    Returns a copy of li_ds with a new variable 'cluster_id' (same length as groups).

    Parameters
    ----------
    li_ds : xr.Dataset
    eps : float
        DBSCAN eps in kilometers (applies to projected x/y). Time dimension is weighted separately.
    time_weight : float
        Multiplier for time to balance against kms in (x,y).
        e.g., time_weight=0.5 means 10 minutes ~ 5 km in distance.
    min_samples : int
        DBSCAN min_samples.
    lat_gap : float
        Gap (deg) to split latitude chunks before clustering.

    Returns
    -------
    xr.Dataset
        A copy of li_ds with new DataArray 'cluster_id' (int64), -1 denotes noise.
    """
    ids  = li_ds["group_id"].values
    lat  = li_ds["latitude"].values
    lon  = li_ds["longitude"].values
    time = li_ds["group_time"].values  # np.datetime64[ns]

    # Convert to minutes relative to first timestamp
    scale_factor_time = 1e9 * 60
    t_minutes = ((time - time.min()).astype("timedelta64[ns]").astype(np.int64)) / scale_factor_time

    df = pd.DataFrame({
        "id": ids,
        "lat": lat,
        "lon": lon,
        "t_scaled": t_minutes * time_weight
    }).sort_values("lat").reset_index(drop=True)

    # Find latitude chunk boundaries
    unique_lats = np.sort(np.unique(df["lat"].values))
    lat_diffs = np.diff(unique_lats)
    gap_idx = np.where(lat_diffs >= lat_gap)[0]
    boundaries = [unique_lats[0], *[unique_lats[i+1] for i in gap_idx], unique_lats[-1] + 1e-12]

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)

    all_parts = []
    label_offset = 0
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        part = df[(df["lat"] >= lo) & (df["lat"] < hi)].copy()
        if part.empty:
            continue

        x_m, y_m = transformer.transform(part["lon"].values, part["lat"].values)
        X = np.stack([np.asarray(x_m) / 1000.0,  # km
                      np.asarray(y_m) / 1000.0,
                      part["t_scaled"].values], axis=1)

        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        labels = db.fit_predict(X)

        # Offset cluster labels (leave noise = -1)
        pos = labels != -1
        labels[pos] += label_offset
        if np.any(pos):
            label_offset = labels[pos].max() + 1

        # reassign clusters with <20 members to noise
        if np.any(pos):
            unique, counts = np.unique(labels[pos], return_counts=True)
            small_clusters = unique[counts < 20]
            if small_clusters.size:
                mask_small = np.isin(labels, small_clusters)
                labels[mask_small] = -1

        part["cluster_id"] = labels
        all_parts.append(part[["id", "cluster_id"]])

    if all_parts:
        labels_all = pd.concat(all_parts, ignore_index=True)
        label_map = dict(zip(labels_all["id"], labels_all["cluster_id"]))
        unique_clusters = np.unique(labels_all["cluster_id"].values)
        n_clusters_total = len(unique_clusters[unique_clusters != -1])
        if n_clusters_total == 0:
            logger.info("No clusters found (all points classified as noise).")
            return None
        else:
            logger.info(f"Total clusters found: {n_clusters_total}")
    else:
        logger.info("No clusters found (no valid points).")
        return None

    # Map back to dataset order
    mapped = pd.Series(ids).map(label_map).fillna(-1).astype("int64").values

    out = li_ds.copy()
    out["cluster_id"] = xr.DataArray(
        mapped,
        dims=li_ds["group_id"].dims,
        attrs={
            "long_name": "Cluster ID",
            "description": f"DBSCAN clustering (eps={eps}, time_weight={time_weight}, min_samples={min_samples}); -1 = noise",
            "units": "1",
        },
    )
    
    # Filter out bad groups and clusters based on quality
    out = filter_clusters_by_quality(out)

    return out


def subcluster_lightning_groups(
    matched_ds: xr.Dataset,
    eps: float = 5.0,
    time_weight: float = 0.5,
    min_samples: int = 20,
) -> xr.Dataset:
    """
    Re-cluster matched lightning groups (where ec_time_diff is valid)
    within each existing cluster_id, using group_time for temporal proximity.

    Adds 'subcluster_id' (float64): NaN = unmatched, -1 = noise.
    """

    # Extract core variables
    lat = matched_ds["latitude"].values
    lon = matched_ds["longitude"].values
    time = matched_ds["group_time"].values
    cluster = matched_ds["cluster_id"].values
    valid = ~np.isnat(matched_ds["ec_time_diff"].values)

    # Initialize with NaN (unmatched)
    sub_ids = np.full(cluster.shape, np.nan, dtype="float64")

    # Transformer for projected coordinates (distance in km)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)

    label_offset = 0
    for cid in np.unique(cluster):
        if cid < 0:
            continue

        # Only consider valid (matched) points in this cluster
        mask = (cluster == cid) & valid
        if not np.any(mask):
            continue

        lat_c = lat[mask]
        lon_c = lon[mask]
        t_c = time[mask]

        # Convert group_time to minutes relative to first timestamp
        t_minutes = ((t_c - t_c.min()).astype("timedelta64[ns]").astype(np.int64)
                     / (1e9 * 60))

        # Project to km + weighted time
        x_m, y_m = transformer.transform(lon_c, lat_c)
        X = np.stack([
            np.asarray(x_m) / 1000.0,
            np.asarray(y_m) / 1000.0,
            t_minutes * time_weight
        ], axis=1)

        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        labels = db.fit_predict(X)

        # Make IDs globally unique (optional, but avoids reuse)
        pos = labels != -1
        labels = labels.astype("float64")  # ensure float for NaN compatibility
        if np.any(pos):
            labels[pos] += label_offset
            label_offset = labels[pos].max() + 1

        sub_ids[mask] = labels

    # Attach to dataset
    out = matched_ds.copy()
    out["subcluster_id"] = xr.DataArray(
        sub_ids,
        dims=matched_ds["group_id"].dims,
        attrs={
            "long_name": "Subcluster ID within parent clusters",
            "description": f"DBSCAN clustering (eps={eps}, time_weight={time_weight}, min_samples={min_samples}); -1 = noise",
            "units": "1",
        },
    )

    return out