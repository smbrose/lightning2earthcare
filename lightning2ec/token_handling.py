import os
from pathlib import Path
import eumdac

# --- Path to credentials.txt --- 
# Assumes it is one directory above the lightning2ec package
CREDENTIALS_FILE = Path(__file__).resolve().parent.parent / "credentials.txt"

def load_credentials(file_path=CREDENTIALS_FILE):
    """Read key-value pairs from a credentials file into a dictionary."""
    creds = {}
    if not file_path.exists():
        raise FileNotFoundError(f"Credentials file not found: {file_path}")
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            creds[key.strip()] = value.strip()
    return creds

# --- EUMETSAT ---
def get_eumetsat_token():
    creds = load_credentials()
    key = creds.get("EUMETSAT_KEY")
    secret = creds.get("EUMETSAT_SECRET")
    if not key or not secret:
        raise ValueError("Missing EUMETSAT_KEY or EUMETSAT_SECRET in credentials file")
    from eumdac import AccessToken, DataStore
    token = AccessToken((key, secret))
    datastore = DataStore(token)
    return token, datastore

# --- ESA MAAP API ---
def get_earthcare_token():
    creds = load_credentials()
    token = creds.get("EARTHCARE_TOKEN")
    if not token:
        raise ValueError("Missing EARTHCARE_TOKEN in credentials file")
    return token
