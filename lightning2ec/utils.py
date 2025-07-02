import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from trollsift import parse

# Pattern to parse EarthCARE filenames. Adjust fields and widths as needed.
_EC_FILENAME_PATTERN = (
    '{product:10}/ECA_{baseline:4}_{product:10}_'
    '{start_time}_{production_time}_{orbit:5}{frame:1}.h5'
)

logger = logging.getLogger(__name__)

def configure_logging(
    level: int = logging.INFO,
    log_file: str = 'lightning2ec.log'
) -> logging.Logger:
    """
    Set up the logging configuration for the project.
    """
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger()


def find_ec_file_pairs(
    date_dir: Path,
    products: List[str],
    frames: List[str]
) -> Dict[str, Dict[str, Path]]:
    """
    Search a dated directory for matching EarthCARE COP/CPR file pairs.

    Args:
        date_dir: Path to the folder named YYYYMMDD containing product subdirs.
        products: List of EarthCARE product names (e.g. ['MSI_COP_2A','CPR_FMR_2A']).
        frames:   List of frame identifiers to include (e.g. ['A','B','D',...]).

    Returns:
        A dict mapping orbit_frame keys (e.g. '12345A') to a dict of
        { product_name: Path(relative_path_to_file) } for complete pairs only.
    """
    result: Dict[str, Dict[str, Path]] = {}

    for product in products:
        prod_dir = date_dir / product
        if not prod_dir.is_dir():
            logger.warning(f"Missing product directory: {prod_dir}")
            continue

        for file_path in prod_dir.iterdir():
            try:
                # Use "product/filename" so parse can match the pattern
                rel = f"{product}/{file_path.name}"
                parsed = parse(_EC_FILENAME_PATTERN, rel)
            except ValueError:
                continue

            orbit = parsed['orbit']
            frame = parsed['frame']
            orbit_frame = f"{orbit}{frame}"

            if frame not in frames or parsed['product'] not in products:
                continue

            entry = result.setdefault(orbit_frame, {})
            entry[parsed['product']] = Path(product) / file_path.name
    # Filter only those with all products
    valid: Dict[str, Dict[str, Path]] = {}
    for of_key, file_map in result.items():
        if set(file_map.keys()) == set(products):
            valid[of_key] = file_map
        else:
            logger.debug(f"Skipping incomplete set {of_key}: found {list(file_map.keys())}")

    return valid


def is_within_li_range(
    lon: np.ndarray,
    times: np.ndarray,
    lon_min: float,
    lon_max: float,
    integration_minutes: int
) -> Tuple[bool, np.datetime64, np.datetime64]:
    """
    Determine if EarthCARE swath overlaps the LI longitude range,
    and compute start/end timestamps for LI download.

    Args:
        lon: 2D array of EarthCARE longitudes.
        times: 1D array of EarthCARE time stamps (numpy.datetime64).
        lon_min, lon_max: bounds for LI matching.
        integration_minutes: half-window around EC times.

    Returns:
        (within_range, start_time, end_time)
    """
    lon_min_ec = float(np.nanmin(lon))
    lon_max_ec = float(np.nanmax(lon))
    logger.info(f"EarthCARE file longitude boundaries: min: {lon_min_ec}, max: {lon_max_ec}")
    within = (
        (lon_min <= lon_min_ec <= lon_max) or
        (lon_min <= lon_max_ec <= lon_max)
    )

    # Compute download window
    start_time = times[0] - np.timedelta64(integration_minutes, 'm')
    end_time   = times[-1] + np.timedelta64(integration_minutes, 'm')

    return within, start_time, end_time
