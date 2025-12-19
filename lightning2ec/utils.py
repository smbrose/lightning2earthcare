import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from trollsift import parse
from collections import defaultdict

# Pattern to parse EarthCARE filenames. Adjust fields and widths as needed.
_EC_FILENAME_PATTERN = (
    '{product:10}/ECA_{baseline:4}_{product:10}_'
    '{start_time}_{production_time}_{orbit:5}{frame:1}.h5'
)

logger = logging.getLogger(__name__)

# --- processing-day context ---
_PROC_CTX = {"day": "-"}

def set_log_day(day_str: str) -> None:
    """Set the current processing day tag (YYYY-MM-DD)."""
    _PROC_CTX["day"] = day_str


class _ProcDayFilter(logging.Filter):
    def filter(self, record):
        record.proc = _PROC_CTX["day"]
        return True


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure the ROOT logger (console only).
    File logging is attached separately per processing month.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Only configure once (avoid duplicate handlers)
    if not root.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [proc=%(proc)s] - %(message)s'
        )

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        sh.addFilter(_ProcDayFilter())
        root.addHandler(sh)

    return root


def set_monthly_log_file(log_dir: str | Path, year: int, month: int) -> None:
    """
    Attach a FileHandler to the ROOT logger that writes to YYYY_MM.log
    based on the *processing date*. Appends if the file already exists.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{year:04d}_{month:02d}.log"
    root = logging.getLogger()

    # Remove existing FileHandler(s) so only one monthly file is active
    for h in list(root.handlers):
        if isinstance(h, logging.FileHandler):
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    fh = logging.FileHandler(log_file, mode="a")
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [proc=%(proc)s] - %(message)s'
    )
    fh.setFormatter(formatter)
    fh.addFilter(_ProcDayFilter())
    root.addHandler(fh)


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

        for file_path in sorted(prod_dir.iterdir()):
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
    for of_key in sorted(result):
        file_map = result[of_key]
        if set(file_map.keys()) == set(products):
            valid[of_key] = {p: file_map[p] for p in products}
        else:
            logger.debug(f"Skipping incomplete set {of_key}: found {list(file_map.keys())}")

    return valid

from .api_utils import query_catalogue, parse_orbit_frame_from_id
def find_ec_file_pairs2(
    products: List[str],
    frames: List[str],
    start_date,
    end_date,
    collection_id="EarthCAREL2Validated_MAAP",
    catalog_url="https://catalog.maap.eo.esa.int/catalogue/",
    asset_key="enclosure_h5",
):
    """
    Build a dict mapping orbit_frame → { product_name: remote_asset_url }.
    Only includes orbits where all requested products are available.
    """
    items = query_catalogue(products, frames, start_date, end_date,
                            collection_id=collection_id,
                            catalog_url=catalog_url)

    tmp = defaultdict(dict)
    for item in items:
        orbit, frame = parse_orbit_frame_from_id(item.id)
        if not orbit or frame not in [f.upper() for f in frames]:
            continue

        orbit_frame = f"{orbit}{frame}"

        # Identify which product type this item corresponds to
        matched_product = next((p for p in products if p in item.id), None)
        if not matched_product:
            continue

        asset = item.assets.get(asset_key)

        if not asset or not asset.href.endswith(".h5"):
            continue

        tmp[orbit_frame][matched_product] = asset.href

    # Only keep entries where all requested products were found
    result = {
        k: v for k, v in tmp.items()
        if set(v.keys()) == set(products)
    }

    # Sort by orbit_frame name
    result = dict(sorted(result.items(), key=lambda x: x[0]))

    logger.info(f"Found {len(result)} complete orbit/frame pairs")
    return result


def _normalize_lon_to_180(lon: np.ndarray) -> np.ndarray:
    """
    Normalize longitudes to [-180, 180] range. Works with 1D/2D arrays.
    """
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0

def _any_overlap_lon(ec_lon_norm: np.ndarray, intervals) -> bool:
    """
    Return True if any EC longitude falls within any of the [lo, hi] intervals.
    `intervals` is an iterable of (lo, hi) in degrees, with lo<=hi, all in [-180,180].
    """
    if ec_lon_norm.size == 0 or np.all(np.isnan(ec_lon_norm)):
        return False
    mask = np.zeros_like(ec_lon_norm, dtype=bool)
    for lo, hi in intervals:
        mask |= (ec_lon_norm >= lo) & (ec_lon_norm <= hi)
    return bool(np.any(mask))

def _glm_east_platform(at_time: np.datetime64) -> str:
    """
    EAST platform: GOES-16 before 2025-04-07, GOES-19 on/after.
    """
    # GLM-East platform handover date (EAST: GOES-16 -> GOES-19)
    _GLM_EAST_SWITCH_DATE = np.datetime64('2025-04-07')
    return 'GOES-16' if at_time < _GLM_EAST_SWITCH_DATE else 'GOES-19'

def _ec_time_window(times: np.ndarray, half_window_minutes: int):
    """
    Compute [start, end] window around EC times.
    """
    start_time = times[0] - np.timedelta64(half_window_minutes, 'm')
    end_time   = times[-1] + np.timedelta64(half_window_minutes, 'm')
    return start_time, end_time

def is_within_satellite_range(
    lon: np.ndarray,
    times: np.ndarray,
    integration_minutes: int = 60,
    allowed_platforms=None,  # e.g. ('MTG-LI','GOES-16') ; None => allow all
):
    """
    Decide which lightning providers to query based on EC longitudes and time.

    Returns a list of dicts (may be empty). Each dict has:
      - source: 'mtg_li' | 'glm_east' | 'glm_west'
      - platform: 'MTG-I1' | 'GOES-16' | 'GOES-19' | 'GOES-18'
      - start_time: np.datetime64
      - end_time: np.datetime64
      - reason: brief string for logging
    """
    selections = []

    if lon.size == 0 or times.size == 0:
        logger.warning("Empty lon/times; no lightning providers selected.")
        return selections

    ec_lon_norm = _normalize_lon_to_180(lon)

    # For concise logging
    try:
        lon_min_ec = float(np.nanmin(ec_lon_norm))
        lon_max_ec = float(np.nanmax(ec_lon_norm))
    except Exception:
        lon_min_ec = lon_max_ec = np.nan

    rep_time = times[len(times)//2]  # midpoint time for platform selection
    start_time, end_time = _ec_time_window(times, integration_minutes)

    # Helper to append if platform is allowed
    def _maybe_add(entry):
        if allowed_platforms is None:
            selections.append(entry)
        else:
            if entry['platform'] in set(allowed_platforms):
                selections.append(entry)
            else:
                logger.info(
                    f"Skipping {entry['source']} ({entry['platform']}) due to platform filter {allowed_platforms}"
                )

    # MTG-LI: lon [-60, 60]
    if _any_overlap_lon(ec_lon_norm, [(-60.0, 60.0)]):
        _maybe_add({
            'source': 'LI',
            'platform': 'MTG-I1',
            'start_time': start_time,
            'end_time': end_time,
            'reason': f"EC lon [{lon_min_ec:.1f},{lon_max_ec:.1f}] overlaps LI lon [-60,60]"
        })

    # ------- GLM: choose the side with larger EC coverage -------
    east_ranges = [(-130.0, -20.0)]
    west_ranges = [(-180.0, -80.0), (170.0, 180.0)]

    valid = np.isfinite(ec_lon_norm)
    # Fraction of EC points within each coverage
    def _in_ranges(arr, ranges):
        m = np.zeros(arr.shape, dtype=bool)
        for lo, hi in ranges:
            m |= (arr >= lo) & (arr <= hi)
        return m

    mask_east = valid & _in_ranges(ec_lon_norm, east_ranges)
    mask_west = valid & _in_ranges(ec_lon_norm, west_ranges)

    denom = float(valid.sum())  # only count valid points
    frac_east = float(mask_east.sum()) / denom
    frac_west = float(mask_west.sum()) / denom

    # Pick only the larger (ties → prefer East)
    if frac_east > 0.0 or frac_west > 0.0:
        if frac_east >= frac_west:
            # GLM-East: platform switches by date
            platform = _glm_east_platform(rep_time)  # 'GOES-16' or 'GOES-19'
            _maybe_add({
                'source': 'GLM',
                'platform': platform,
                'start_time': start_time,
                'end_time': end_time,
                'reason': (f"EC lon [{lon_min_ec:.1f},{lon_max_ec:.1f}] overlaps GLM-East lon [-130,-20]; "
                           f"selected East (coverage {frac_east:.2%}) with platform {platform}")
            })
        else:
            # GLM-West: fixed GOES-18
            _maybe_add({
                'source': 'GLM',
                'platform': 'GOES-18',
                'start_time': start_time,
                'end_time': end_time,
                'reason': (f"EC lon [{lon_min_ec:.1f},{lon_max_ec:.1f}] overlaps GLM-West lon [-180,-80]∪[170,180]; "
                           f"selected West (coverage {frac_west:.2%}) with platform GOES-18")
            })

    if selections:
        for s in selections:
            logger.info(f"Selected {s['source']} ({s['platform']}): {s['start_time']} -> {s['end_time']} | {s['reason']}")
    else:
        logger.info(f"No lightning coverage for EC lon [{lon_min_ec:.1f},{lon_max_ec:.1f}].")

    return selections