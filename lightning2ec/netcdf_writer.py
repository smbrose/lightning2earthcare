import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
import xarray as xr
import numpy as np

logger = logging.getLogger(__name__)


def make_li_output_path(
    base_dir: Path,
    li_times: Sequence[np.datetime64],
    orbit_frame: str,
    close_count: int
) -> Path:
    """
    Construct an output NetCDF file path based on Lightning match times and orbit frame.

    Args:
        base_dir: Directory where outputs are stored.
        li_times: Sequence of numpy.datetime64 times of matched events in time.
        orbit_frame: Identifier for orbit and frame (e.g. '12345A').
        close_count: Number of events within the nadir distance threshold.

    Returns:
        A Path to the new NetCDF file, with directories created as needed.
    """
    # Convert numpy times to pandas DatetimeIndex for easy formatting
    times = pd.to_datetime(li_times)
    start_str = times.min().strftime('%Y%m%dT%H%M%SZ')
    end_str   = times.max().strftime('%Y%m%dT%H%M%SZ')
    count_all = len(times)

    # Directory per start date
    date_dir = base_dir #/ times.min().strftime('%Y%m%d')
    date_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"LI_LGR_{start_str}_{end_str}_{count_all}_"
        f"{close_count}_{orbit_frame}.nc"
    )
    return date_dir / filename


def write_netcdf(
    ds: xr.Dataset,
    output_path: Path
) -> None:
    """
    Save an xarray Dataset to a NetCDF file.

    Args:
        ds: The Dataset to write.
        output_path: Destination file path.
    """
    try:
        ds.to_netcdf(path=str(output_path))
        logger.info(f"Saved NetCDF to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save NetCDF {output_path}: {e}")
        raise
