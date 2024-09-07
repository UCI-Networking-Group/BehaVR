
## BehaVR Adversary Training and Evaluation

This folder contains code for feature engineering and feature selection, comparing data processing methods (FBL and FBA), optimizing and training models, designing and evaluating BehaVR adversaries, analyzing top features, etc.

## System Requirements
We tested the code on a server with the following configuration (reference hardware):

- CPU: Intel Xeon Silver 4316 (2 sockets x 20 cores x 2 threads)
- Memory: 512 GiB
- GPU: 2x NVIDIA RTX A5000 (24 GiB of video memory each)
- OS: Debian GNU/Linux 12 (Bookworm)

A GPU is not required to run the machine learning models; however, we highly recommend using one to speed up the process.

## Environment Setup

All the code in this repository is written in Python, and we use Conda to manage the Python dependencies.
Before getting started, please install Conda by following the [installation-instructions](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).

Create a new conda environment named `behavr` using python 3.10 and install the necessary dependencies. 
Please see the [requirement.txt](https://github.com/UCI-Networking-Group/BehaVR/blob/main/Adversary/requirement.txt) file for all the dependencies.
```console
$ conda create --name behavr python=3.10
$ conda activate behavr
$ pip install -r requirements.txt
```

To navigate to the `Adversary` folder inside the `BehaVR` repository using the command line, you would use the following commands:

```console
$ cd Adversary
```

## Usage

- **Base Script:** `
- `~/run.py`: Automatically runs the main file and adjusts for different adversarial scenarios, like app adversary, open-world, or zero-day scenarios.
- **Code:**
  - `~/Input_data.py`: Contains functions for loading and preprocessing input data from different sensor groups: BM (Body Motion), EG (Eye Gaze), HJ (Hand Joint), and FE (Facial Expression).
  - `~/model.py`: Contains necessary functions for tuning models, loading necessary models depending on the sensor group, and finding top features.
  - `~/preprocess.py`: Contains necessary functions for data preprocessing, feature engineering, loading train/test data, and adversarial conditions.
  - `~/main.py`: Main file that generates output results depending on the type of adversaries. `main.py` contains all the variable parameters to run the model in different settings.
  - `~/plot.py`: This script includes code for plotting evaluation graphs using body motion data. The code can be easily adapted to other sensor groups.
  - `~/sample_plot.py`: Plots some sample graphs that can be found in the paper. We used saved results to plot those graphs.

To run the code, please run the commands on your terminal based on the type of BehaVR adversary and sensor group. For example, to run an app adversary model on the Facial Expression (FE) sensor group on the first 15 apps, where your chosen model is Random Forest, run the following command in the terminal:

```console
$ cd code
$ python main.py --SG='BM' --num_app=15 --adv='App' --Model='RF'
```
Or, you can simply change your command in the run.py and run the following command in the terminal:

```console
$ python run.py
```
