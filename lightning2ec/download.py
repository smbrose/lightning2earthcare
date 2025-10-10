import os
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import numpy as np
import eumdac
from fnmatch import fnmatch

logger = logging.getLogger(__name__)

_COLLECTION_IDS = {
    'lightning_events_filtered': 'EO:EUM:DAT:0690',
    'lightning_groups':           'EO:EUM:DAT:0782',
}

from .token_handling import get_eumetsat_token

_TOKEN, _DATASTORE = get_eumetsat_token()

def download_li(
    start_time,
    end_time,
    collection_name: str,
    output_dir: Path
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
    if collection_name not in _COLLECTION_IDS:
        msg = f"Collection '{collection_name}' is not defined."
        logger.error(msg)
        raise ValueError(msg)

    dt_start = _to_datetime(start_time)
    dt_end   = _to_datetime(end_time)

    coll_id = _COLLECTION_IDS[collection_name]
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
