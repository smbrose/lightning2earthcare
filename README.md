# lightning2earthcare

`lightning2earthcare` is a pipeline to collocate **EarthCARE MSI/CPR** data with **MTG/GOES Lightning observations** over a date range. The pipeline now supports fetching EarthCARE data directly from the remote ESA MAAP STAC endpoint instead of a local directory

---

## Getting Started

Build environment file:

```bash
conda env create -f environment.yaml
```
To activate the environment after creation, use: 

```bash
conda activate Lightning2ec
```

## Usage

Example command:

```bash
python -m lightning2ec.cli \
    --lightning-dir "PATH_TO_YOUR_LIGHTNING_DATA" \
    --start-date 2025-08-22 \
    --end-date 2025-08-22 \
    --lightning-platform MTG-I1

```
By default, the pipeline processes both MTG and GOES lightning data, so specifying `--lightning-platform` is optional.

## Credentials

All API keys and tokens are stored in a `credentials.txt` file located **one directory above** the `lightning2ec` repository.
**New**: Use the long-lasting token that can be generated here: https://portal.maap.eo.esa.int/ini/services/auth/token/90dToken.php 


### Creating `credentials.txt`

1. Create a plain text file called `credentials.txt`.
2. Populate it with your keys and tokens in this format:
```
EUMETSAT_KEY=your_eumetsat_key_here
EUMETSAT_SECRET=your_eumetsat_secret_here
CLIENT_ID=offline-token
CLIENT_SECRET=p1eL7uonXs6MDxtGbgKdPVRAmnGxHpVE
OFFLINE_TOKEN=your_earthcare_longlasting_token_here
```
### Important

- Do **not** use quotes around the values.  
- Ensure there are no trailing spaces or extra characters.  

## Recent Updates

### Credentials Handling
- [24/10] Long-lasting token with refresh option replaces the former EARTHCARE_TOKEN 
- Centralized reading of `EUMETSAT_KEY`, `EUMETSAT_SECRET`, and `EARTHCARE_TOKEN`.  
- No hardcoded tokens in scripts.  
- This is all handled in the new `token_handling.py` script.

### Remote URLs for EarthCARE Data
- Pipeline now queries the ESA MAAP STAC catalog.  
- `find_ec_file_pairs2()` replaced the old local-directory version at least for EarthCARE data.   
- MSI and CPR data are fetched via HTTPS using `fsspec`. 

### New `api_utils.py` Script
- Contains `query_catalogue()`, `parse_orbit_frame()`, and `fetch_earthcare_data()` for handling STAC queries and remote dataset access.  

### Pipeline Adjustments
- CLI now works without `--ec-base`.  
- `find_ec_file_pairs2()` automatically finds complete orbit/frame pairs from STAC query results.  
- `prepare_ec2()` replaces `prepare_ec()` and works with remote MSI URLs.  
- `build_cpr_summary2()` replaced `build_cpr_summary()` and works with remote CPR datasets. 