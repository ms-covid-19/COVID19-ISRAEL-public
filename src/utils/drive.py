"""A convenience utility for getting files from Google Drive."""
import os
import pickle

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

DATA_DIR = os.path.join(os.path.dirname(__file__),
                        '..', '..', 'data', 'Drive')
CREDS_FILE = os.path.join(DATA_DIR, 'credentials.json')
TOKEN_FILE = os.path.join(DATA_DIR, 'token.pickle')

# If modifying these scopes, delete the file token.pickle.
# TODO: This scope is too permissive. Should switch to 'drive.file' once I
#  figure out how to use it.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

_client = None


def _init():
    """Initializes the connection to Google Drive."""
    global _client
    if _client is not None:
        return

    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            print('No login token found.')
            print('Opening browser. In the consent screen, '
                  'give the app permissions in order to continue.')
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)

    _client = build('drive', 'v3', credentials=creds)


def get_sheet_csv(file_id: str, out_path: str):
    """Downloads a Google Sheet as CSV."""
    _init()
    request = _client.files().export_media(fileId=file_id,
                                           mimeType='text/csv')
    with open(out_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            # print("Download %d%%." % int(status.progress() * 100))


def get_file(file_id: str, out_path: str):
    """Downloads a general data file from Drive."""
    _init()
    request = _client.files().get_media(fileId=file_id)
    with open(out_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            # print("Download %d%%." % int(status.progress() * 100))


if __name__ == '__main__':
    sheet_id = '1uN75gAJV8SV4s7HCdJreGun4fE0xFOmfpj7fhNd5BJI'
    csv_id = '1xBwaL5A6c1XTv0dFV_2aoT3zTwrSTHZ-'
    get_sheet_csv(sheet_id, os.path.join(DATA_DIR, 'test_sheet.csv'))
    get_file(csv_id, os.path.join(DATA_DIR, 'test_file.csv'))
