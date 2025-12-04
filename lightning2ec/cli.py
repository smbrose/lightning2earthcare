import click
from datetime import timedelta
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

from .lightning_pipeline import process_one_source
from .preprocess import prepare_ec2
from .utils import find_ec_file_pairs2, is_within_satellite_range, configure_logging, set_monthly_log_file, set_log_day

@click.command()

@click.option(
    '--lightning-dir', 'lightning_base_path',
    type=click.Path(file_okay=False),
    required=True,
    help="Base directory for Lightning downloads and outputs"
)
@click.option(
    '--log-dir',
    type=click.Path(file_okay=False),
    default='logs',
    show_default=True,
    help="Directory where monthly logs are written"
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
    '--integration', 'integration_minutes',
    default=60,
    show_default=True,
    help="Half-window of LI integration in minutes"
)
@click.option(
    '--lightning-platform', 'lightning_platforms',
    multiple=True,
    type=click.Choice(['MTG-I1', 'GOES-16', 'GOES-18', 'GOES-19']),
    default=['MTG-I1', 'GOES-16', 'GOES-18', 'GOES-19'],
    show_default=True,
    help="Lightning platforms to include"
)
@click.option(
    '--distance-threshold', 'distance_threshold_km',
    type=float,
    default=5,
    show_default=True,
    help="Distance threshold from CPR track in kilometers"
)
@click.option(
    '--time-threshold', 'time_threshold_s',
    type=int,
    default=300,
    show_default=True,
    help="Temporal threshold from CPR acquisition in seconds"
)

def run_pipeline(
    lightning_base_path,
    log_dir,
    start_date, end_date,
    products, frames,
    integration_minutes,
    lightning_platforms,
    distance_threshold_km, time_threshold_s
):
    """Run the EarthCARE + Lightning collocation pipeline over a date range."""
    logger = configure_logging()
    l_base = Path(lightning_base_path)
    current_date = start_date
    current_month = None

    while current_date <= end_date:
        # set processing-day tag
        set_log_day(f"{current_date:%Y-%m-%d}")
        month_key = (current_date.year, current_date.month)
        if month_key != current_month:
            # Switch log file when month changes (or first iteration)
            set_monthly_log_file(log_dir, current_date.year, current_date.month)
            current_month = month_key

        logger.info(f"Processing date: {current_date:%Y-%m-%d}")
        # NEW: find EC file pairs remotely via STAC, not local files
        try:
            pairs = find_ec_file_pairs2(
                products=products,
                frames=frames,
                start_date=current_date,
                end_date=current_date
            )
        # NEW: Deal wihh potential STAC query errors
        except Exception as e:
            logger.error(f"STAC query failed for {current_date:%Y-%m-%d}: {e}")
            current_date += timedelta(days=1)
            continue

        if not pairs:
            logger.info(f"No matching EarthCARE orbits found for {current_date:%Y-%m-%d}")
            current_date += timedelta(days=1)
            continue

        for orbit_frame, file_map in pairs.items():
            logger.info(f"Processing orbit frame: {orbit_frame}")

            # NEW: These are now remote URLs (strings)
            msi_url = file_map[products[0]]
            cpr_url = file_map[products[1]]

            # NEW: URLs directly into prepare_ec2()
            ec_lon, ec_lat, ec_cth, ec_times = prepare_ec2(msi_url)

            selections = is_within_satellite_range(ec_lon, ec_times, integration_minutes,
                                                   allowed_platforms=tuple(lightning_platforms))
            if not selections:
                logger.info(f"{orbit_frame}: outside all lightning coverages, skipping")
                continue

            for sel in selections:
                source_key = sel['source']      # 'mtg_li' | 'glm_east' | 'glm_west'
                platform   = sel['platform']    # 'MTG-I1' | 'GOES-16' | 'GOES-19' | 'GOES-18'
                t0         = sel['start_time']
                t1         = sel['end_time']

                logger.info(f"{orbit_frame}: processing {source_key} ({platform}) {t0} → {t1}")
                process_one_source(
                    source_key, platform, 
                    t0, t1,
                    l_base,
                    orbit_frame,
                    ec_lon, ec_lat, ec_cth, ec_times,
                    cpr_url,
                    distance_threshold_km, time_threshold_s
                )

        current_date += timedelta(days=1)

if __name__ == '__main__':
    run_pipeline()