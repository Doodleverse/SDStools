# How to Install

Currently SDS_tools runs in the coastseg environment. Follow the CoastSeg installation [guide](https://satelliteshorelines.github.io/CoastSeg/basic-install-guide/) to set up a conda environment with CoastSeg installed then in this activate environment run any of the SDS_tools scripts.

# Example

Clone the github repo

```
git clone --depth 1 https://github.com/Doodleverse/SDStools.git
```

## Install Additional Dependencies

SDS tools differs from CoastSeg in that it uses 3 additional dependencies

- `statsmodel`: Provides statistical models and filters, such as the Hampel Filter.
- `bathyreq `: Allows downloading topobathy data using [BathyReq](https://github.com/NeptuneProjects/BathyReq)
- `cdsapi`: Enables downloading ERA5 wave data from the [CDSAPI](https://pypi.org/project/cdsapi/)

```
conda activate coastseg
cd <location SDS_tools installed>
pip install statsmodel bathyreq cdsapi
```

### How to Use the Scripts

After installing the dependencies, activate the CoastSeg environment, navigate to the scripts directory within SDS_tools, and run the desired script.

```
conda activate coastseg
cd <location SDS_tools installed>
cd scripts
python <script_name.py>

```

Replace <script_name.py> with the name of the script you want to run.
