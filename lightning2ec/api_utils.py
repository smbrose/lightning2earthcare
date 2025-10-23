import fsspec
import xarray as xr
import os
from pystac_client import Client  # Ensure pystac-client is installed
import requests
import logging
from datetime import date
from collections import defaultdict
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .token_handling import get_earthcare_token

def query_catalogue(
    products: List[str],
    frames: List[str],
    start_date: date,
    end_date: date,
    collection_id: str = "EarthCAREL2Validated_MAAP",
    catalog_url: str = "https://catalog.maap.eo.esa.int/catalogue/",
):
    """
    Query ESA MAAP STAC catalogue for specified products, frames, and date range.
    Returns a list of pystac.Items.
    """
    client = Client.open(catalog_url)
    datetime_str = [f"{start_date.strftime('%Y-%m-%dT00:00:00Z')}",
            f"{end_date.strftime('%Y-%m-%dT23:59:59Z')}"]


    # Build filter: (productType='X' OR productType='Y') AND (frame='A' OR frame='B')
    product_filter = " OR ".join([f"productType = '{p}'" for p in products])
    frame_filter = " OR ".join([f"frame = '{f}'" for f in frames])
    combined_filter = f"({product_filter}) AND ({frame_filter})"

    logger.info(f"Querying STAC:\n  products={products}\n  frames={frames}\n  date={datetime_str}")
    logger.debug(f"  Combined filter: {combined_filter}")

    search = client.search(
        collections=[collection_id],
        datetime=datetime_str,
        bbox = [-180, -60, 60, 60] or [170, -60, 180, 60],
        filter=combined_filter,
        method="GET",
    )

    items = list(search.items())
    ba_items = [item for item in items if item.id.startswith("ECA_EXBA")]
    logger.info(f"STAC returned {len(items)} matching items.")
    return ba_items


def parse_orbit_frame_from_id(item_id: str):
    """Extract orbit+frame from final underscore field in STAC item id."""
    try:
        last = item_id.split("_")[-1]
        return last[:-1], last[-1].upper()
    except Exception:
        return None, None

def fetch_earthcare_data(ds_url, group="ScienceData"):
    """
    Fetch EarthCARE data from a remote HTTPS URL and return it as an xarray.Dataset.

    Args:
        ds_url (str): HTTPS URL to the dataset (STAC asset)
        group (str): NetCDF group to open (default: "ScienceData")

    Returns:
        xarray.Dataset: Loaded dataset.
    """
    if not ds_url.startswith("http"):
        raise ValueError(f"Only remote HTTPS URLs are supported. Got: {ds_url}")

    logger.info(f"Opening remote EarthCARE dataset via HTTPS: {ds_url}")

    token = get_earthcare_token()

    fs = fsspec.filesystem("https", headers={"Authorization": f"Bearer {token}"})
    with fs.open(ds_url, "rb") as f:
        ds = xr.open_dataset(f, engine="h5netcdf", group=group)
        ds.load()  # load into memory

    logger.info(f"Successfully loaded dataset: {ds_url}")
    return ds