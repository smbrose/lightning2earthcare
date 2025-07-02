import click
from datetime import timedelta
from pathlib import Path

from .download import download_li
from .preprocess import prepare_ec
from .utils import find_ec_file_pairs, is_within_li_range, configure_logging
from .parallax import apply_parallax_shift
from .collocation import (
    merge_li_datasets,
    match_li_to_ec,
    compute_nadir_distances
)
from .netcdf_writer import make_li_output_path, write_netcdf

logger = configure_logging()

@click.command()
@click.option(
    '--ec-base', 'ec_base_path',
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Root directory for EarthCARE HDF5 files"
)
@click.option(
    '--lightning-dir', 'li_base_path',
    type=click.Path(file_okay=False),
    required=True,
    help="Base directory for Lightning downloads and outputs"
)
@click.option(
    '--start-date',
    type=click.DateTime(formats=['%Y-%m-%d']),
    required=True,
    help="Start date (YYYY-MM-DD)"
)
@click.option(
    '--end-date',
    type=click.DateTime(formats=['%Y-%m-%d']),
    required=True,
    help="End date (YYYY-MM-DD)"
)
@click.option(
    '--product', 'products',
    multiple=True,
    default=['MSI_COP_2A', 'CPR_FMR_2A'],
    show_default=True,
    help="EarthCARE products to process"
)
@click.option(
    '--frame', 'frames',
    multiple=True,
    default=['A', 'B', 'D', 'E', 'F', 'H'],
    show_default=True,
    help="Orbit frames to include"
)
@click.option(
    '--li-collection',
    default='lightning_groups',
    show_default=True,
    help="EUMETSAT DS collection name for Lightning"
)
@click.option(
    '--lon-min',
    default=-80,
    show_default=True,
    help="Minimum longitude for Lightning matching"
)
@click.option(
    '--lon-max',
    default=80,
    show_default=True,
    help="Maximum longitude for Lightning matching"
)
@click.option(
    '--integration', 'integration_minutes',
    default=30,
    show_default=True,
    help="Half-window of LI integration in minutes"
)
@click.option(
    '--sat-lon', 'satellite_lon',
    default=0,
    show_default=True,
    help="Satellite longitude"
)
@click.option(
    '--sat-lat', 'satellite_lat',
    default=0,
    show_default=True,
    help="Satellite latitude"
)
@click.option(
    '--sat-alt', 'satellite_alt',
    default=35786400,
    show_default=True,
    help="Satellite altitude (meters)"
)
@click.option(
    '--distance-threshold', 'distance_threshold_km',
    type=float,
    default=2.5,
    show_default=True,
    help="Distance threshold from CPR track in kilometers"
)
@click.option(
    '--time-threshold', 'time_threshold_s',
    type=int,
    default=150,
    show_default=True,
    help="Temporal threshold from CPR acquisition in seconds"
)

def run_pipeline(
    ec_base_path, li_base_path,
    start_date, end_date,
    products, frames,
    li_collection,
    lon_min, lon_max,
    integration_minutes,
    satellite_lon, satellite_lat, satellite_alt,
    distance_threshold_km, time_threshold_s
):
    """Run the EarthCARE + Lightning collocation pipeline over a date range."""
    ec_base = Path(ec_base_path)
    li_base = Path(li_base_path)

    current_date = start_date
    while current_date <= end_date:
        logger.info(f"Processing date: {current_date:%Y-%m-%d}")
        date_dir = ec_base / current_date.strftime('%Y%m%d')
        pairs = find_ec_file_pairs(date_dir, products, frames)

        for orbit_frame, file_map in pairs.items():
            cop_file = date_dir / file_map[products[0]]
            cpr_file = date_dir / file_map[products[1]]

            lon, lat, cth, ec_times = prepare_ec(cop_file)
            within, li_start, li_end = is_within_li_range(
                lon, ec_times, lon_min, lon_max, integration_minutes
            )
            if not within:
                logger.info(f"{orbit_frame}: outside longitude bounds, skipping")
                continue

            li_paths = download_li(
                li_start, li_end, li_collection, li_base
            )
            merged_li = merge_li_datasets(li_paths)
            shifted_lat, shifted_lon = apply_parallax_shift(
                lon, lat, cth,
                satellite_lon, satellite_lat, satellite_alt
            )

            matched_ds, matched_times = match_li_to_ec(
                merged_li, cth, ec_times,
                shifted_lat, shifted_lon,
                satellite_lon, satellite_lat,
                satellite_alt, integration_minutes
            )
            if matched_ds is None:
                logger.info(f"{orbit_frame}: no lightning matches found")
                continue

            final_ds, close_count = compute_nadir_distances(
                matched_ds, cpr_file,
                distance_threshold_km=distance_threshold_km,
                time_threshold_s=time_threshold_s
            )
            output_file = make_li_output_path(
                li_base, matched_times, orbit_frame, close_count
            )
            write_netcdf(final_ds, output_file)

        current_date += timedelta(days=1)

if __name__ == '__main__':
    run_pipeline()
