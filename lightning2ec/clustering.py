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
    Drops flag and auxiliary variables after filtering.

    Keeps only groups where:
    - All flag values are 0
    - Belong to a cluster that has <= `cluster_bad_threshold` proportion of bad groups

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 'cluster_id' and flag variables.
    cluster_bad_threshold : float
        Threshold above which clusters are considered bad and removed.

    Returns
    -------
    xr.Dataset
        Cleaned dataset with only high-quality groups and clusters.
    """
    cluster_id_var = "cluster_id"
    flag_vars = [
        "l1b_missing_warning",
        "l1b_geolocation_warning",
        "l1b_radiometric_warning"
    ]
    auxiliary_vars = [
        "auxiliary_dataset_identifier",
        "auxiliary_dataset_status",
        "group_filter_qa"
    ]

    # Create per-group mask
    quality_mask = np.ones(ds.sizes["groups"], dtype=bool)
    for flag in flag_vars:
        quality_mask &= ds[flag].values == 0

    cluster_ids = ds[cluster_id_var].values
    df = ds[[cluster_id_var]].to_dataframe().reset_index()
    df["is_bad"] = ~quality_mask
    df = df[df[cluster_id_var] != -1]

    # Evaluate per-cluster bad group ratio
    bad_rates = df.groupby(cluster_id_var)["is_bad"].mean()
    bad_clusters = bad_rates[bad_rates > cluster_bad_threshold].index.values

    is_in_bad_cluster = np.isin(cluster_ids, bad_clusters)
    final_mask = (~is_in_bad_cluster) & quality_mask
    logger.info(f"Dropped {len(bad_clusters)} of {bad_rates.size} clusters for quality issues.")

    # Subset dataset
    ds = ds.isel(groups=final_mask)

    # Drop flag and auxiliary variables
    vars_to_drop = [v for v in flag_vars + auxiliary_vars if v in ds.variables]
    ds = ds.drop_vars(vars_to_drop)

    return ds


def cluster_li_groups(
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
