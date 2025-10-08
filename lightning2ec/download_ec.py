from pystac_client import Client 
import fsspec
import requests 
from tqdm import tqdm


def download_file_with_bearer_token(url, token, disable_bar=False):
  """
  Downloads a file from a given URL using a Bearer token.
  """

  try:
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes
    file_size = int(response.headers.get('content-length', 0))

    chunk_size = 8 * 1024 * 1024 # Byes - 1MiB
    file_path = url.rsplit('/', 1)[-1] 
    print(file_path)
    with open(file_path, "wb") as f, tqdm(
        desc=file_path,
        total=file_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
        disable=disable_bar,
      ) as bar:
      for chunk in response.iter_content(chunk_size=chunk_size):
        read_size=f.write(chunk)
        bar.update(read_size)

    if (disable_bar): 
      print(f"File downloaded successfully to {file_path}")

  except requests.exceptions.RequestException as e:
    print(f"Error downloading file: {e}")



def download_assets_from_items(items, token, asset_key='enclosure_1', disable_bar=False):
    for item in items:
        if asset_key in item.assets:
            url = item.assets[asset_key].href
            download_file_with_bearer_token(url, token, disable_bar)
        else:
            print(f"Asset '{asset_key}' not found in item: {item.id}")


# -------------------------------
# Main Script
# -------------------------------
if __name__ == "__main__":

    catalog_url = 'https://catalog.maap.eo.esa.int/catalogue/'
    catalog = Client.open(catalog_url)
    collection_ID = ['EarthCAREL2Validated_MAAP']

    search = catalog.search(
        collections=collection_ID, 
        filter="productType = 'MSI_COP_2A' or productType = 'CPR_FMR_2A' ", # Filter by product type
        datetime = ['2025-08-20T00:00:00Z', '2025-08-21T00:00:00Z'] ,
        method = 'GET', 
        #max_items=2 
        )

    items = list(search.items())
    token= 'eyJhbGciOiJSUzI1NiIsInR5cCIgOiAiSldUIiwia2lkIiA6ICJQXzJqUU50Y3QtOGR2cW1qVG5QWDVNc3BfT1Zid2lzVFlHbmFwM2tUWVdJIn0.eyJleHAiOjE3NTkxNzM4NjYsImlhdCI6MTc1OTEzMDY2NywianRpIjoiYjhlZWI5ZGItODM1Yi00MThhLWI1N2UtMzYwZmQ4N2Q0ZmVkIiwiaXNzIjoiaHR0cHM6Ly9pYW0ubWFhcC5lby5lc2EuaW50L3JlYWxtcy9lc2EtbWFhcCIsInN1YiI6ImY3YzViMDRjLTZhNTQtNDU0Mi04OTNmLTRkNTgzOTUzNGJmMiIsInR5cCI6IkJlYXJlciIsImF6cCI6ImVzYS1tYWFwLXBvcnRhbCIsInNpZCI6IjNmNDBkYjRiLWE5NjQtNDdlNi05OTBkLTJmMWI5NDQxNDc3OSIsInNjb3BlIjoib3BlbmlkIiwiY291bnRyeSI6IkRFIiwic291cmNlcyI6WyJFT1A6RVNBOk1BQVAiLCJFT1A6RVNBOkVBUlRILU9OTElORS1OT1QtUkVBRFkiXSwiT2EtU2lnbmVkLVRjcyI6IkJJT01BU1NfQ09NTUlTU0lPTklORyxCSU9NQVNTX0NPUkVfQ09NTSxBRU9MVVNfT1BFTixNRVJJU19GUkVFX0FORF9PUEVOLFNNT1NfRlJFRV9BTkRfT1BFTixFU0Ffb3Blbl9hbmRfZnJlZSxUUE1fT1BFTixHT0NFX09QRU4iLCJncm91cHMiOlsiL2NvbGxlY3Rpb25zL0JJT01BU1MtUERHUy9CaW9tYXNzQXV4IiwiL2NvbGxlY3Rpb25zL0JJT01BU1MtUERHUy9CaW9tYXNzQXV4SU9DIiwiL2NvbGxlY3Rpb25zL0JJT01BU1MtUERHUy9CaW9tYXNzQXV4UmVzdCIsIi9jb2xsZWN0aW9ucy9CSU9NQVNTLVBER1MvQmlvbWFzc0xldmVsMElPQyIsIi9jb2xsZWN0aW9ucy9CSU9NQVNTLVBER1MvQmlvbWFzc0xldmVsMWFJT0MiLCIvY29sbGVjdGlvbnMvQklPTUFTUy1QREdTL0Jpb21hc3NMZXZlbDFiSU9DIiwiL2NvbGxlY3Rpb25zL0JJT01BU1MtUERHUy9CaW9tYXNzTGV2ZWwxY0lPQyIsIi9jb2xsZWN0aW9ucy9CSU9NQVNTLVBER1MvQmlvbWFzc0xldmVsMmFJT0MiLCIvY29sbGVjdGlvbnMvQklPTUFTUy1QREdTL0Jpb21hc3NMZXZlbDJiSU9DIiwiL0FwcGxpY2F0aW9uL0VjbGlwc2UtY2hlIiwiL2luaXRpYXRpdmVzL2VhcnRoY2FyZS9yb2xlcy9kZXZlbG9wZXIiLCIvaW5pdGlhdGl2ZXMvYmlvbWFzcy9yb2xlcy9kZXZlbG9wZXIiLCIvQXBwbGljYXRpb24vUEFMLU5leHRDbG91ZC9lYXNocmRhdGEiXSwiZWR1UGVyc29uVW5pcXVJZCI6InlnemN1S041LTBlNzJNQWlXZ2ZuTGc9PUBlc2EtYWRmcy5pbnQiLCJwcmVmZXJyZWRfdXNlcm5hbWUiOiJzYXNraWEuYnJvc2VAZXNhLmludCIsImVtYWlsIjoic2Fza2lhLmJyb3NlQGVzYS5pbnQifQ.noG7jSMMO3L5_nra9o_7CuUHLBdiP-MeYe32Vuvr57w0jYB9rgdc0bK09nUSfDhd76KZYYHU8ZbEYB9WKfmMGvH0hlBhohm7M3sBVdW6dvkVPJBqET-kHmTcQ2NwQc6ZTe5FK2IXLFLe8bMoWIfo8fluyzfHL4V4tFlw2DiC1vyb1W8As8vPkmqJEeBSDahsdvloVQzH7fnZDsRRjXLhrRJ_0-MXNCEcroaB9umRSfi4WcBbv35B3d28qltLW9_qpJeQ9e0KMkmpjAQzxnTTsPeyaBgrqWAboa6TN9iJHhVklQWBGlNifR0Ps7WTXJcyZarJ3yV8y2p_W615lZENRw'
    download_assets_from_items(items, token)
