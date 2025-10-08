import os
import logging
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import numpy as np
import eumdac

logger = logging.getLogger(__name__)

_COLLECTION_IDS = {
    'lightning_events_filtered': 'EO:EUM:DAT:0690',
    'lightning_groups':           'EO:EUM:DAT:0782',
}

_EUMETSAT_KEY = 'LfI3E05VvcUTd1e5WSIey4fKBJUa'
_EUMETSAT_SECRET = 'pIH4odTz6hq6ckZM5SAb4Dg29jAa'
# Read credentials from environment
#_EUMETSAT_KEY = os.getenv('EUMETSAT_CONSUMER_KEY')
#_EUMETSAT_SECRET = os.getenv('EUMETSAT_CONSUMER_SECRET')

#if not (_EUMETSAT_KEY and _EUMETSAT_SECRET):
#    logger.error("EUMETSAT_CONSUMER_KEY and/or EUMETSAT_CONSUMER_SECRET not set in environment")
#    raise RuntimeError("Missing EUMETSAT API credentials")

# Initialize access token and datastore
_TOKEN = eumdac.AccessToken((_EUMETSAT_KEY, _EUMETSAT_SECRET))
_DATASTORE = eumdac.DataStore(_TOKEN)

def download_li(
    start_time,
    end_time,
    collection_name: str,
    output_dir: Path
) -> list[Path]:
    """
    Download and unzip Lightning data from EUMETSAT Data Store.

    Args:
        start_time: numpy.datetime64 or datetime start of range
        end_time:   numpy.datetime64 or datetime end of range
        collection_name: key in _COLLECTION_IDS
        output_dir: base Path to save downloads and extracted folders

    Returns:
        List of Paths to extracted directories containing LI files.
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
        try:
            with product.open() as src:
                zip_path = output_dir / src.name
                extract_dir = zip_path.with_suffix('')
                print("DEBUG extract_dir:", extract_dir)

                # If already extracted (folder exists), reuse it
                if extract_dir.exists():
                    logger.info(f"Data already extracted in {extract_dir}, skipping")
                    target_dirs.append(extract_dir)
                    continue

                # Download zip archive if missing
                if not zip_path.exists():
                    with zip_path.open('wb') as dst:
                        shutil.copyfileobj(src, dst)
                    logger.info(f"Downloaded {zip_path.name}")

                # Ensure Windows uses long path syntax
                def to_long_path(path: Path) -> str:
                    """Return Windows long-path form if on Windows, else normal path."""
                    p = str(path)
                    if os.name == "nt":  # Windows only
                        if not p.startswith("\\\\?\\"):
                            p = "\\\\?\\" + os.path.abspath(p)
                    return p

                try:
                    with zipfile.ZipFile(str(zip_path), 'r') as zf:
                        for member in zf.namelist():
                            target = extract_dir / member
                            target.parent.mkdir(parents=True, exist_ok=True)

                            # Open inside-zip file and write to disk with long-path support
                            with zf.open(member) as source, open(to_long_path(target), "wb") as dst:
                                shutil.copyfileobj(source, dst)

                    zip_path.unlink()
                    logger.info(f"Extracted and removed {zip_path.name}")
                    target_dirs.append(extract_dir)
                except Exception as e:
                    logger.error(f"Failed to extract {zip_path.name}: {e}")


                # Extract and remove archive
        #         try:
        #             with zipfile.ZipFile(zip_path, 'r') as zf:
        #                 zf.extractall(path=extract_dir)
        #             zip_path.unlink()
        #             logger.info(f"Extracted and removed {zip_path.name}")
        #             target_dirs.append(extract_dir)
        #         except Exception as e:
        #             logger.error(f"Failed to extract {zip_path.name}: {e}")
        except Exception as e:
            logger.error(f"Error handling product {product}: {e}")

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
