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
    Cluster lightning groups in space-time using DBSCAN, after splitting into latitude chunks.
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
    MAX_GROUPS_PER_CHUNK = 150_000

    ids   = li_ds["group_id"].values
    f_ids = li_ds["flash_id"].values
    lat   = li_ds["latitude"].values
    lon   = li_ds["longitude"].values
    time  = li_ds["group_time"].values  # np.datetime64[ns]

    N = len(ids)
    
    # Convert to minutes relative to first timestamp
    scale_factor_time = 1e9 * 60
    t_minutes = ((time - time.min()).astype("timedelta64[ns]").astype(np.int64)) / scale_factor_time
    t_scaled = t_minutes * float(time_weight)

    # --- Sort everything by latitude once ---
    idx_sort = np.argsort(lat)
    ids_s    = ids[idx_sort]
    f_ids_s  = f_ids[idx_sort]
    lat_s    = lat[idx_sort]
    lon_s    = lon[idx_sort]
    t_s      = t_scaled[idx_sort]

    # --- Compute latitude chunk boundaries ---
    unique_lats = np.unique(lat_s)
    lat_diffs = np.diff(unique_lats)
    gap_idx = np.where(lat_diffs >= lat_gap)[0]
    boundaries = [unique_lats[0], *[unique_lats[i+1] for i in gap_idx], unique_lats[-1] + 1e-12]

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:6933", always_xy=True)
    rng = np.random.default_rng()

    # Output arrays in *sorted* index space
    cluster_id_sorted = np.full(N, -1, dtype=np.int64)
    sampled_sorted = np.zeros(N, dtype=bool)  # True if this point was part of the sample

    label_offset = 0

    # --- Process each latitude chunk ---
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        # indices in sorted space belonging to this chunk
        mask_chunk = (lat_s >= lo) & (lat_s < hi)
        chunk_idx = np.where(mask_chunk)[0]
        n_chunk = chunk_idx.size
        if n_chunk == 0:
            continue

        # --- Build sampling mask per chunk (in local chunk index space) ---
        if n_chunk > MAX_GROUPS_PER_CHUNK:
            # local mask for this chunk: length = n_chunk
            sampled_chunk = np.zeros(n_chunk, dtype=bool)

            f_chunk = f_ids_s[chunk_idx]
            # Build mapping flash_id -> list of local indices within the chunk
            flash_to_positions = {}
            for local_pos, fid in enumerate(f_chunk):
                if fid in flash_to_positions:
                    flash_to_positions[fid].append(local_pos)
                else:
                    flash_to_positions[fid] = [local_pos]

            n_flashes_chunk = len(flash_to_positions)

            if n_flashes_chunk > MAX_GROUPS_PER_CHUNK:
                print(
                    f"Warning: chunk [{lo:.4f}, {hi:.4f}) has {n_flashes_chunk} flashes "
                    f"> MAX_GROUPS_PER_CHUNK={MAX_GROUPS_PER_CHUNK}. "
                    "Sampling one per flash (sample will exceed limit)."
                )
                max_target = n_flashes_chunk
            else:
                max_target = MAX_GROUPS_PER_CHUNK

            # One representative per flash
            for fid, positions in flash_to_positions.items():
                positions = np.asarray(positions, dtype=int)
                chosen_local = rng.choice(positions)
                sampled_chunk[chosen_local] = True

            current_sampled = int(sampled_chunk.sum())
            remaining_capacity = max_target - current_sampled

            if remaining_capacity > 0:
                unsampled_local = np.where(~sampled_chunk)[0]
                if unsampled_local.size > 0:
                    if remaining_capacity < unsampled_local.size:
                        extra_local = rng.choice(
                            unsampled_local, size=remaining_capacity, replace=False
                        )
                    else:
                        extra_local = unsampled_local
                    sampled_chunk[extra_local] = True
        else:
            # Small chunk: all points sampled
            sampled_chunk = np.ones(n_chunk, dtype=bool)

        # Mark sampled points in global sorted space
        sampled_idx_global = chunk_idx[sampled_chunk]
        sampled_sorted[sampled_idx_global] = True

        # --- Prepare sampled subset for DBSCAN ---
        if sampled_idx_global.size == 0:
            # no points to cluster in this chunk
            continue

        lat_sp = lat_s[sampled_idx_global]
        lon_sp = lon_s[sampled_idx_global]
        t_sp   = t_s[sampled_idx_global]

        # Drop rows with NaNs / inf in lat, lon, t before projection
        finite0 = (
            np.isfinite(lat_sp) &
            np.isfinite(lon_sp) &
            np.isfinite(t_sp)
        )
        if not finite0.all():
            lat_sp = lat_sp[finite0]
            lon_sp = lon_sp[finite0]
            t_sp   = t_sp[finite0]
            sampled_idx_global = sampled_idx_global[finite0]

        if lat_sp.size == 0:
            continue

        x_m, y_m = transformer.transform(lon_sp, lat_sp)
        X = np.stack(
            [np.asarray(x_m) / 1000.0,
             np.asarray(y_m) / 1000.0,
             t_sp],
            axis=1,
        )

        # Drop any rows with NaN/inf produced by projection
        finite = np.isfinite(X).all(axis=1)
        if not finite.all():
            X = X[finite]
            sampled_idx_global = sampled_idx_global[finite]

        if X.shape[0] == 0:
            continue

        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        labels = db.fit_predict(X)  # shape (n_valid,)

        # Offset cluster labels (leave noise = -1)
        pos = labels != -1
        labels_offset = labels.copy()
        labels_offset[pos] += label_offset
        if np.any(pos):
            label_offset = labels_offset[pos].max() + 1

        # reassign clusters with <20 members to noise
        if np.any(pos):
            unique_lbl, counts_lbl = np.unique(labels_offset[pos], return_counts=True)
            small_clusters = unique_lbl[counts_lbl < 20]
            if small_clusters.size:
                mask_small = np.isin(labels_offset, small_clusters)
                labels_offset[mask_small] = -1

        # Write labels into cluster_id_sorted for these sampled, valid rows
        cluster_id_sorted[sampled_idx_global] = labels_offset

    # --- Majority-vote for unsampled points (in sorted space) ---
    # First, build majority cluster per flash_id, using only sampled & non-noise points
    from collections import defaultdict

    flash_clusters = defaultdict(list)
    valid_sampled_mask = sampled_sorted & (cluster_id_sorted != -1)

    for idx in np.where(valid_sampled_mask)[0]:
        fid = f_ids_s[idx]
        cid = int(cluster_id_sorted[idx])
        flash_clusters[fid].append(cid)

    majority_map = {}
    for fid, cids in flash_clusters.items():
        if not cids:
            continue
        vals, counts = np.unique(cids, return_counts=True)
        majority_map[fid] = int(vals[np.argmax(counts)])

    # Assign unsampled points to flash majority cluster if available
    unsampled_mask = (~sampled_sorted)
    for idx in np.where(unsampled_mask)[0]:
        fid = f_ids_s[idx]
        if fid in majority_map:
            cluster_id_sorted[idx] = majority_map[fid]

    # --- Map back from sorted order to original dataset order ---
    cluster_id_orig = np.empty(N, dtype=np.int64)
    cluster_id_orig[idx_sort] = cluster_id_sorted

    unique_clusters = np.unique(cluster_id_orig)
    n_clusters_total = int(np.sum(unique_clusters != -1))
    if n_clusters_total == 0:
        logger.info("No clusters found (all points classified as noise).")
        return None
    else:
        logger.info(f"Total clusters found: {n_clusters_total}")

    # Attach cluster_id to dataset
    out = li_ds.copy()
    out["cluster_id"] = xr.DataArray(
        cluster_id_orig,
        dims=li_ds["group_id"].dims,
        attrs={
            "long_name": "Cluster ID",
            "description": f"DBSCAN clustering (eps={eps}, time_weight={time_weight}, min_samples={min_samples}); -1 = noise",
        },
    )

    # Quality filtering as before
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
        },
    )

    return out