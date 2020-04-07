# Zips a data file using the key, places it in the github-data directory and
# tracks it with git-LFS.
#
# USE ONLY THIS SCRIPT TO ADD FILES TO THE GITHUB-DATA FOLDER.

set -e

if [ "$1" == "" ]; then
  echo "Usage:"
  echo "zip_raw_data.sh FILE"
  echo
  echo "FILE should be relative to the data directory."
  echo "For example 'Raw/forms/COVID-19-Bot-0104.csv'."
  exit 1
fi
if [ "$2" != "" ]; then
  echo "Please use with one file at a time."
  exit 2
fi

# cd to data directory.
cd "$(git rev-parse --show-toplevel)"
cd data

# Make sure input file and key exist.
if [ ! -f "$1" ]; then
  echo "Could not find input file data/$1."
  exit 2
fi
if [ ! -f "zip_secret_key.txt" ]; then
  echo "Could not find zip_secret_key.txt."
  exit 2
fi

# Work.
base="$(basename $1)"
zip -P "$(cat zip_secret_key.txt)" "$base.zip" "$1"
mv "$base.zip" ../github-data/
cd ..
git lfs track "github-data/$base.zip"
