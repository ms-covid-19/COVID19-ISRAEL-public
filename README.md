# COVID19-ISRAEL-public

This is the public repository of the COVID19-ISRAEL survey project.

Initial work is available in [MedRxiv](https://www.medrxiv.org/content/10.1101/2020.03.19.20038844v1). 

Notice that in order to make code run you must add survey data to `data` folder according to its README.md file.

## Setup

### Pip only (without conda)

1. pip install -r pip-requirments.txt
2. download libspatialindex binaries into "[your-env-dir]/site-packages/rtree":
  https://github.com/ms-covid-19/COVID19-ISRAEL-public/releases/download/dep/spatialindex-64.dll
  https://github.com/ms-covid-19/COVID19-ISRAEL-public/releases/download/dep/spatialindex_c.dll