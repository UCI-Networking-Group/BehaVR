
## BehaVR Adversary Training and Evaluation

This folder contains the essential code for feature engineering and selection, comparing data processing methods (FBL and FBA), optimizing and training models for different adversaries, designing and evaluating various BehaVR adversary settings, and analyzing top features (see Sections 4.1.3-5 in the main paper).


## Getting Started
To navigate to the `Adversary` folder inside the `BehaVR` repository using the command line, you would use the following command:

```console
$ cd Adversary
```

## System Requirements
We tested the code on a server with the following configuration (reference hardware):

- CPU: Intel Xeon Silver 4316 (2 sockets x 20 cores x 2 threads)
- Memory: 512 GiB
- GPU: 2x NVIDIA RTX A5000 (24 GiB of video memory each)
- OS: Debian GNU/Linux 12 (Bookworm)

A GPU is not required to run the machine learning models, however, we highly recommend using one to speed up the process.

## Environment Setup

All the code in this repository is written in Python, and we use Conda to manage the Python dependencies.
Before getting started, please install Conda by following the [installation-instructions](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).

Create a new conda environment named `behavr` and install the necessary dependencies: 

```console
$ conda env create -n behavr -f environment.yml
$ conda activate behavr
```

## Usage

- **Base Script:** `
  - `~/run.py`: Automatically runs the main file and adjusts for different adversarial scenarios, like app adversary, open-world, or zero-day scenarios. Please edit the files for corresponding adversarial settings.
  
- **Code:**
  - `~/Input_data.py`: Contains functions for loading and preprocessing input data from different sensor groups: BM (Body Motion), EG (Eye Gaze), HJ (Hand Joint), and FE (Facial Expression).
  - `~/model.py`: Contains necessary functions for tuning models, loading necessary models depending on the sensor group, and finding top features.
  - `~/preprocess.py`: Contains necessary functions for data preprocessing, feature engineering, loading train/test data, and adversarial conditions.
  - `~/main.py`: Main file that generates output results depending on the type of adversaries. `main.py` contains all the variable parameters to run the model in different settings.
  - `~/plot.py`: This script includes code for plotting evaluation graphs using body motion data. The code can be easily adapted to other sensor groups.
  - `~/sample_plot.py`: Plots some sample graphs that can be found in the paper. We used saved results to plot those graphs.

- **Adversarial arguments:**
In the terminal, when running `main.py`, you can use the following arguments. These arguments depend on the adversarial settings, but are optional in a sense that the `main.py` can automatically execute with default settings.

```
--help                  Show this help message and exit.
--SG                    The sensor group-BM/FE/EG/HJ.
--adv                   Type of Adversary- App:App Adversary, 
                        emotion:Identification using facial emotion, 
                        Sensor_fusion:Sensor Group Model Ensemble, 
                        OW:Open World Setting, Zero-Day:Zero-Day Settings.
--n_emo                 The number of emotions we consider.
--SG1                   The first sensor group for Ensemble multiple sensors-BM/FE/EG/HJ.
--SG2                   The second sensor group for Ensemble multiple sensors-BM/FE/EG/HJ.
--data_process          FBA or FBL.
--feature_elim          Eliminating top features (Figure 9 Evaluation).   
--OW                    If open-world setting is true.
--num_user              Total number of users.
--rt                    remove user id x ir rt=t, all users will be used if rt=f.
--num_app               Total number of apps.
--Model                 Model type, RF=Random Forest, XGB=Xboost.
--target                Target Classifier.
--cross_val             Cross validation value.
--ratio                 Block length controller ratio.
--r1                    Block length controller ratio for SG1.
--r2                    Block length controller ratio for SG2.
--M                     Controlling sub-session time, M inversely proportional to subsession.
--f_n                   How many top feature we want to observe.
--output_dir            Directory to save output results.

```
## Running Code
To run the code, please run the commands on your terminal and change the arguments based on the type of BehaVR adversary and sensor group:

```console
$ cd code
$ python main.py --argument1 --argument2
```
Or, you can simply change your command in the run.py and run the following command in the terminal:

```console
$ python run.py
```

## Examples
- **Example 1**: To evaluate the identification accuracy for the Body Motion (BM) sensor group and app adversary using the Random Forest model, with results from 20 apps and 20 users (refer to Section 5.1 in the main paper), run the following command:

```console
$ python main.py --SG='BM' --num_app=20  --num_user=20 --adv='App' --Model='RF'
```
Alternatively, to use the Facial Expression sensor group (Section 5.4 in the main paper), run:

```console
$ python main.py --SG='FE' --num_app=20  --num_user=20 --adv='App' --Model='RF'
```

- **Example 2**: To evaluate identification based on facial emotions with 20 users using the Random Forest model (refer to Sections 5.4.1 in the main paper), run the following command:

```console
$ python main.py --SG='BM' --num_app=20  --num_user=20 --adv='emotion' --Model='RF'
```

- **Example 3**: To evaluate Zero-day scenarios with 20 users and the Facial Expression (FE) sensor group using the Random Forest model (refer to Sections 4.2 and 5.5.2 in the main paper), run the following command:

```console
$ python main.py --SG='FE' --num_user=20 --adv='Zero-Day' --Model='RF'
```

Alternatively, to use the XGBoost algorithm, run:

```console
$ python main.py --SG='FE' --num_user=20 --adv='Zero-Day' --Model='XGB'
```

- **Example 4**: To ensemble the Hand Joint (HJ) and Eye Gaze (EG) sensor group models using the Random Forest algorithm (refer to Section 5.5.3 in the main paper), run the following command:

```console
$ python main.py --SG1='HJ' --SG2='EG' --num_user=20 --adv='Sensor_fusion' --Model='RF'
```

Alternatively, to use the XGBoost algorithm, run:

```console
$ python main.py --SG1='HJ' --SG2='EG' --num_user=20 --adv='Sensor_fusion' --Model='XGB'
```
