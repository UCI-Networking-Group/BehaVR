

## BehaVR Data Processing and Abstraction

This section demonstrates how the raw sensor data has been processed to convert time series data into feature blocks, laying the groundwork for subsequent feature engineering. This folder includs files reorganization, sensor data processing and sensor data abstraction.

## System Requirements
We run the code on a PC with the following configuration:

- CPU: Intel i7-8550U (8-core, 1.80-2.0 GHz)
- Memory: 16.0 GiB
- OS: Windows 10 Pro 64-bits

## Environment Setup

All the code in this repository is written in Python, and we use Conda to manage the Python dependencies.
Create a new conda environment named `behavr-data` and install the necessary dependencies: 

```console
$ conda env create -n behavr-data -f environment.yml
$ conda activate behavr-data
```

## Usage  
  
- **Code:**
  - `~/folder-reorganization.ipynb`: Reorganizes the data files, the result is each folder contains one sensor group.
  - `~/PROCESSING`: Contains 8 ipynb files, with 2 block division method (`FBA`-Fixed Block Amount, `FBL`-Fixed Block Length)and 4 different sensor groups(`SG11`-Body Motion Head, `SG12`-Body Motion Hand, `SG2`-Eye Gaze, `SG3`-Hand Joints, `SG4`-Facial Expression). 
  - `~/plot-duration-variety-bp.ipynb`: Generates sensor data duration overview, Fig.3 (a) & (b). Notes: we use random numbers to show the functionality of plotting code.
  
- **Data Directory:**
  - When readers want to use the code, please create the following data directory. 
  - `~/data/raw`: Stores raw time-series, which includes sensor data per use per app per round(sub-folder name: `uid_aid_rid`).
  - `~/data/reorg`: Stores raw time-series, which includes sensor data per sensor group(sub-folder name: `SG_id`).
  - `~/data/clean`: Stores cleaned sensor data(sub-folder name: `SG_id`).
  - `~/data/result`: Stores results after block division and abstraction.
  
