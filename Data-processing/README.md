

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
 

## Experiments 
- **Experiment 1**: To process and abstract Eye Gaze(EG) data using FBA method, please go through following steps:
  - Create data directory and put the raw time series into `~/data/raw` with sub-folder name: `uid_aid_rid`, then run `folder-reorganization.ipynb`. Raw Eye Gaze data from all data collection will be in `~/data/reorg/SG2`.
  - Open `FBN-SG2.ipynb` and set the parameter `r`(block ratio) and `game_id_list`(app id that involved in the data collection), then run `FBN-SG2.ipynb`.
  -  Cleaned sensor data will appear in `~/data/clean/SG2` and data abstraction results will appear in `~/data/result`.

- **Experiment 2**: To process and abstract Body Motion(BM) data using FBA method, please go through following steps:

  - Create data directory and put the raw time series into `~/data/raw` with sub-folder name: `uid_aid_rid`, then run `folder-reorganization.ipynb`. Raw Body Motion data from all data collection will be in `~/data/reorg/SG11`(BM-Head) and `~/data/reorg/SG12`(BM-Hand) .
  -  Open `FBN-SG11.ipynb` and `FBN-SG12.ipynb`, set the parameter `r`(block ratio) and `game_id_list`(app id that involved in the data collection), then run `FBN-SG11.ipynb` first and `FBN-SG12.ipynb` second.
  - Cleaned sensor data will appear in `~/data/clean/SG11` and `~/data/clean/SG12`, data abstraction results will appear in `~/data/result`.

- **Experiment 3**: To process and abstract Body Motion(BM) data using FBL method, please go through following steps:

  - Create data directory and put the raw time series into `~/data/raw` with sub-folder name: `uid_aid_rid`, then run `folder-reorganization.ipynb`. Raw Body Motion data from all data collection will be in `~/data/reorg/SG11`(BM-Head) and `~/data/reorg/SG12`(BM-Hand) .
  -  Run `FBA-SG11.ipynb` and `FBA-SG12.ipynb` data cleaning related codes, then cleaned data will appear in `~/data/clean/SG11` and `~/data/clean/SG12`.
  -  Open `FBL-SG11.ipynb` and `FBL-SG12.ipynb`, set the parameter `TS`(time slot) , then run `FBL-SG11.ipynb` first and `FBL-SG12.ipynb` second. Data abstraction results will appear in `~/data/result`.
