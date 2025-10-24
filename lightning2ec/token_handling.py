import os
from pathlib import Path
import eumdac
import requests
import json
import base64

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
def get_earthcare_token_old():
    creds = load_credentials()
    token = creds.get("EARTHCARE_TOKEN")
    if not token:
        raise ValueError("Missing EARTHCARE_TOKEN in credentials file")
    return token

def get_earthcare_token():
    """Use OFFLINE_TOKEN to fetch a short-lived access token."""
    creds = load_credentials()

    OFFLINE_TOKEN = creds.get("OFFLINE_TOKEN")
    CLIENT_ID = creds.get("CLIENT_ID")
    CLIENT_SECRET = creds.get("CLIENT_SECRET")

    if not all([OFFLINE_TOKEN, CLIENT_ID, CLIENT_SECRET]):
        raise ValueError("Missing OFFLINE_TOKEN, CLIENT_ID, or CLIENT_SECRET in credentials file")

    url = "https://iam.maap.eo.esa.int/realms/esa-maap/protocol/openid-connect/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": OFFLINE_TOKEN,
        "scope": "offline_access openid"
    }

    response = requests.post(url, data=data)
    response.raise_for_status()

    response_json = response.json()
    access_token = response_json.get('access_token')

    if not access_token:
        raise RuntimeError("Failed to retrieve access token from IAM response")

    return access_token
