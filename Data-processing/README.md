

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

- **Folder Reorganization Script:**
  - `./folder-reorganization.ipynb`: Reorganizes the data files so that each folder contains one sensor group.

- **Block Division and Plot:**
  - `/FBA`: Contains 5 Jupyter notebooks using the Fixed Block Amount (FBA) method for different sensor groups:
    - `FBA-SG11.ipynb`: FBA method for Head (Body Motion sensor group).
    - `FBA-SG12.ipynb`: FBA method for Controllers (Body Motion sensor group).
    - `FBA-SG2.ipynb`: FBA method for Eye Gaze sensor group.
    - `FBA-SG3.ipynb`: FBA method for Hand Joints sensor group.
    - `FBA-SG4.ipynb`: FBA method for Facial Expression sensor group.
  
  - `/FBL`: Contains 5 Jupyter notebooks using the Fixed Block Length (FBL) method for different sensor groups:
    - `FBL-SG11.ipynb`: FBL method for Head (Body Motion sensor group).
    - `FBL-SG12.ipynb`: FBL method for Controllers (Body Motion sensor group).
    - `FBL-SG2.ipynb`: FBL method for Eye Gaze sensor group.
    - `FBL-SG3.ipynb`: FBL method for Hand Joints sensor group.
    - `FBL-SG4.ipynb`: FBL method for Facial Expression sensor group.

  - `./plot-duration-variety-bp.ipynb`: Generates sensor data duration overview (Figure 3(a) & 3(b)). Note: Random numbers are used to show the functionality of the plotting code.

- **Data Directory:**
  - Please create the following data directories for proper organization:
    - `/data/raw`: Stores raw time-series data (sub-folder name: `uid_aid_rid` for each user, app, and round).
    - `/data/reorg`: Stores reorganized time-series data by sensor group (sub-folder name: `SG_id`).
    - `/data/clean`: Stores cleaned sensor data (sub-folder name: `SG_id`).
    - `/data/result`: Stores results after block division and abstraction.


## Experiments 
- **Experiment 1**: To process and abstract Eye Gaze(EG) data using FBA method, please go through following steps:
  - Create data directory and put the raw time series into `/data/raw` with sub-folder name: `uid_aid_rid`, then run `folder-reorganization.ipynb`. Raw Eye Gaze data from all data collection will be in `/data/reorg/SG2`.
  - Open `FBN-SG2.ipynb` and set the parameter `r`(block ratio) and `game_id_list`(app id that involved in the data collection), then run `FBN-SG2.ipynb`.
  -  Cleaned sensor data will appear in `/data/clean/SG2` and data abstraction results will appear in `/data/result`.

- **Experiment 2**: To process and abstract Body Motion(BM) data using FBA method, please go through following steps:

  - Create data directory and put the raw time series into `/data/raw` with sub-folder name: `uid_aid_rid`, then run `folder-reorganization.ipynb`. Raw Body Motion data from all data collection will be in `/data/reorg/SG11`(BM-Head) and `/data/reorg/SG12`(BM-Hand) .
  -  Open `FBN-SG11.ipynb` and `FBN-SG12.ipynb`, set the parameter `r`(block ratio) and `game_id_list`(app id that involved in the data collection), then run `FBN-SG11.ipynb` first and `FBN-SG12.ipynb` second.
  - Cleaned sensor data will appear in `/data/clean/SG11` and `/data/clean/SG12`, data abstraction results will appear in `/data/result`.

- **Experiment 3**: To process and abstract Body Motion(BM) data using FBL method, please go through following steps:

  - Create data directory and put the raw time series into `/data/raw` with sub-folder name: `uid_aid_rid`, then run `folder-reorganization.ipynb`. Raw Body Motion data from all data collection will be in `/data/reorg/SG11`(BM-Head) and `/data/reorg/SG12`(BM-Hand) .
  -  Run `FBA-SG11.ipynb` and `FBA-SG12.ipynb` data cleaning related codes, then cleaned data will appear in `/data/clean/SG11` and `/data/clean/SG12`.
  -  Open `FBL-SG11.ipynb` and `FBL-SG12.ipynb`, set the parameter `TS`(time slot) , then run `FBL-SG11.ipynb` first and `FBL-SG12.ipynb` second. Data abstraction results will appear in `/data/result`.
