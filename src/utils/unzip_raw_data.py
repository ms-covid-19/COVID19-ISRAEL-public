"""Unzips the files in the github-data directory into data."""
import os
from glob import glob
from zipfile import ZipFile

from config import GITHUB_DATA_DIR, DATA_DIR

out_dir = DATA_DIR
files = glob(os.path.join(GITHUB_DATA_DIR, '*.zip'))
pwd = open(os.path.join(DATA_DIR, 'zip_secret_key.txt'), 'rb').read().strip()

for f in files:
    print('Extracting', f)
    with ZipFile(f, 'r') as z:
        z.extractall(out_dir, pwd=pwd)
