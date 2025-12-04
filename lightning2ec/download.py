import os
import re
import logging
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List
import numpy as np
import xarray as xr
import eumdac
from fnmatch import fnmatch
from concurrent.futures import ThreadPoolExecutor
import warnings

logger = logging.getLogger(__name__)

from .token_handling import get_eumetsat_token

_TOKEN, _DATASTORE = get_eumetsat_token()
_COLLECTION_ID = {
    'lightning_groups':           'EO:EUM:DAT:0782',
}

def download_li(
    start_time,
    end_time,
    output_dir: Path,
    collection_name='lightning_groups'
) -> list[Path]:
    """
    Download Lightning data from EUMETSAT Data Store.

    Args:
        start_time: numpy.datetime64 or datetime start of range
        end_time:   numpy.datetime64 or datetime end of range
        collection_name: key in _COLLECTION_IDS
        output_dir: base Path to save downloads

    Returns:
        List of Paths to LI files.
    """
    dt_start = _to_datetime(start_time)
    dt_end   = _to_datetime(end_time)

    coll_id = _COLLECTION_ID[collection_name]
    collection = _DATASTORE.get_collection(coll_id)
    products = collection.search(dtstart=dt_start, dtend=dt_end)
    logger.info(f"Found {products.total_results} products in '{collection_name}'")
    if products.total_results == 0:
        logger.warning("Skipping download")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    target_dirs: List[Path] = []

    for product in products:
        entries = list(product.entries)
        matches = [e for e in entries if fnmatch(e, "*BODY*.nc")]
        if not matches:
            logger.warning(f"No *BODY*.nc entry found for {product.id}")
            continue

        for entry in matches:
            try:
                with product.open(entry=entry) as src:
                    nc_path = output_dir / src.name

                    # If data already exist locally, reuse it
                    if nc_path.exists():
                        logger.info(f"Data already exist, skipping: {nc_path.name}")
                        target_dirs.append(nc_path)
                        continue

                    # Download data if missing
                    with nc_path.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    logger.info(f"Downloaded {nc_path.name}")
                    target_dirs.append(nc_path)

            except Exception as e:
                logger.error(f"Error handling product {product.id} / entry {entry}: {e}")

    return target_dirs


def _to_datetime(ts) -> datetime:
    """
    Convert numpy.datetime64 or datetime to a timezone-aware datetime.
    """
    if isinstance(ts, np.datetime64):
        seconds = ts.astype('datetime64[s]').astype(int)
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    if isinstance(ts, datetime):
        return ts
    raise TypeError(f"Unsupported type for timestamp: {type(ts)}")


_BUCKET_BY_PLATFORM = {
    'GOES-16': 'noaa-goes16',
    'GOES-18': 'noaa-goes18',
    'GOES-19': 'noaa-goes19',
}

_GLM_PRODUCT = 'GLM-L2-LCFA'  # Full-disk GLM L2 (LCFA) product

# Example: OR_GLM-L2-LCFA_G16_s20232400000000_e20232400000200_c20232400000223.nc
_GLM_NAME_RE = re.compile(
    r'^OR_GLM-L2-LCFA_G(?P<plat>\d{2})_s(?P<s>\d{14})_e(?P<e>\d{14})_c(?P<c>\d{14})\.nc$'
)

def _parse_glm_timefield(s: str) -> datetime:
    # Take only the first 13 digits (YYYYJJJHHMMSS)
    dt = datetime.strptime(s[:13], '%Y%j%H%M%S').replace(tzinfo=timezone.utc)
    return dt

def _iter_hours(dt_start: datetime, dt_end: datetime):
    """Yield (year, jday, hour) covering [dt_start, dt_end] inclusive."""
    # Floor to hour for start; ceil to hour for end
    cur = dt_start.replace(minute=0, second=0, microsecond=0)
    if cur < dt_start:
        cur = cur  # already floored
    end = (dt_end.replace(minute=0, second=0, microsecond=0)
           + (timedelta(hours=1) if dt_end.minute or dt_end.second or dt_end.microsecond else timedelta(0)))
    while cur <= end:
        yield cur.year, int(cur.strftime('%j')), cur.hour
        cur += timedelta(hours=1)
    

def load_merge_glm(
    start_time,
    end_time,
    platform: str,
    product: str = _GLM_PRODUCT,
    chunksize: int = 10000, # native chunksize 256, but the performance is better with larger chunks
    max_workers: int = 8,
) -> xr.Dataset | None:
    """
    Open GOES GLM L2 (LCFA) NetCDFs directly from NOAA S3 for a time window
    and return a single merged xarray.Dataset.
    """

    try:
        import s3fs
    except ImportError:
        logger.error("s3fs is required to read GLM from NOAA S3. Install with: pip install s3fs")
        return None

    # ---- platform validation ----
    if platform not in _BUCKET_BY_PLATFORM:
        logger.error(f"Unsupported platform {platform!r}. Expected one of {list(_BUCKET_BY_PLATFORM)}")
        return None

    # ---- normalize times ----
    dt_start = _to_datetime(start_time)
    dt_end   = _to_datetime(end_time)
    if dt_end < dt_start:
        logger.warning("load_merge_glm: end_time < start_time; nothing to do.")
        return None

    # ---- S3 listing / key filtering ----
    bucket = _BUCKET_BY_PLATFORM[platform]
    fs = s3fs.S3FileSystem(anon=True)

    prefixes = [
        f"{bucket}/{product}/{year:04d}/{jday:03d}/{hour:02d}/"
        for (year, jday, hour) in _iter_hours(dt_start, dt_end)
    ]

    found_keys = []
    for pref in prefixes:
        try:
            keys = fs.ls(pref)
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.error(f"Error listing {pref}: {e}")
            continue

        for key in keys:
            fname = key.split('/')[-1]
            m = _GLM_NAME_RE.match(fname)
            if not m:
                continue

            s_dt = _parse_glm_timefield(m.group('s'))
            e_dt = _parse_glm_timefield(m.group('e'))

            # keep only files that overlap requested time window
            if not (e_dt < dt_start or s_dt > dt_end):
                found_keys.append(key)

    if not found_keys:
        logger.info(f"GLM: no files found in {bucket}/{product} for {dt_start} → {dt_end}")
        return None

    found_keys = sorted(set(found_keys))
    urls = [f"s3://{key}" for key in found_keys]

    # ---- helper: open+preprocess ONE file ----
    def _open_and_preprocess_one(url: str) -> xr.Dataset | None:
        """
        Open a single GLM file lazily and apply cleanup + latitude clip.
        Returns None if nothing remains after clipping.
        """
        try:
            # Let xarray choose sensible chunks based on the store
            ds = xr.open_dataset(
                fs.open(url, mode='rb'),
                engine='h5netcdf',
                chunks="auto",   # or omit 'chunks' entirely
            )
        except Exception:
            ds = xr.open_dataset(
                fs.open(url, mode='rb'),
                engine='netcdf4',
                chunks="auto",
            )

        # NOW rechunk along groups AFTER loading
        if 'number_of_groups' in ds.dims:
            ds = ds.chunk({'number_of_groups': chunksize})
            ds = ds.rename({'number_of_groups': 'groups'})
        elif 'groups' in ds.dims:
            ds = ds.chunk({'groups': chunksize})

        # 3. promote coords with 'groups' dim to data vars
        coord_names = [c for c in ds.coords if 'groups' in getattr(ds[c], 'dims', ())]
        if coord_names:
            ds = ds.reset_coords(names=coord_names, drop=False)

        # 4. drop variables that are NOT along 'groups'
        drop_names = [
            name for name in ds.variables
            if 'groups' not in getattr(ds[name], 'dims', ())
        ]
        ds = ds.drop_vars(drop_names, errors='ignore')

        # 5. drop specific unwanted vars
        ds = ds.drop_vars(['group_time_offset', 'group_area'], errors='ignore')

        # 6. rename to nicer variable names
        rename_map = {
            'group_frame_time_offset': 'group_time',
            'group_lat': 'latitude',
            'group_lon': 'longitude',
            'group_energy': 'radiance',
            'group_parent_flash_id': 'flash_id',
        }
        present = {k: v for k, v in rename_map.items() if k in ds.variables}
        if present:
            ds = ds.rename(present)

        return ds

    # ---- open/preprocess all files (parallel, order preserved) ----
    datasets: list[xr.Dataset] = []
    def _safe_open(url):
        try:
            return _open_and_preprocess_one(url)
        except Exception as e:
            logger.error(f"Failed to open/preprocess {url}: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for ds_part in ex.map(_safe_open, urls):
            if ds_part is not None:
                datasets.append(ds_part)

    if not datasets:
        logger.error("No GLM datasets had groups after filtering.")
        return None

    # ---- concat using YOUR original concat config ----
    try:
        merged = xr.concat(
            datasets,
            dim='groups',
            data_vars='minimal',
            coords='minimal',
            join='outer',
            combine_attrs='drop_conflicts',
            compat='override',
        )
        
        merged = merged.assign_attrs({"platform": platform})
        logger.info(f"Merged {len(datasets)} GLM files for platform {platform}")
        return merged

    except Exception as e:
        logger.error(f"Failed to concatenate GLM datasets: {e}")
        return None