
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

### Estimated Time and Storage Consumption

Without a GPU, all experiments required approximately 35-40 hours of runtime. With the GPU set up as we described in the system requirements, the runtime may range between 8-10 hours.



## Environment Setup

All the code in this repository is written in Python, and we use Conda to manage the Python dependencies.
Before getting started, please install Conda by following the [installation-instructions](https://conda.io/projects/conda/en/latest/user-guide/install/linux.html).

Create a new conda environment named `behavr` and install the necessary dependencies: 

```console
$ conda env create -n behavr -f environment.yml
$ conda activate behavr
```

### Testing the Environment 
For testing the environment, please run the following command in the console: 
```console
$ ./env-test.sh
```

## Usage

- **Base Script:**
  - `./run.py`: Automatically runs the main file and adjusts for different adversarial scenarios, like app adversary, open-world, or zero-day scenarios. Please edit the files for corresponding adversarial settings.

- **Code:**
  - `code/`
    - `Input_data.py`: Contains functions for loading and preprocessing input data from different sensor groups: BM (Body Motion), EG (Eye Gaze), HJ (Hand Joint), and FE (Facial Expression).
    - `model.py`: Contains necessary functions for tuning models, loading necessary models depending on the sensor group, and finding top features.
    - `preprocess.py`: Contains necessary functions for data preprocessing, feature engineering, loading train/test data, and adversarial conditions.
    - `main.py`: Main file that generates output results depending on the type of adversaries. `main.py` contains all the variable parameters to run the model in different settings.
    - `plot.py`: This script includes code for plotting evaluation graphs. The code can be easily adapted to any sensor group.
    - `sample_plot.py`: Plots some sample graphs that can be found in the paper. We used saved results to plot those graphs.

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

## Main Results and Claims

In Section 5, we detail user identification and related evaluations based on various adversarial settings. Specifically:

- User Identification with Sensor Groups, including Body Motion, Eye Gaze, Hand Joints, and Facial Expressions for the App adversary (Sections 5.1–5.4). This will be illustrated in Experiment 1.
- User identification via Facial Emotion Expression, discussed in Section 5.4.1 and this will be illustrated in Experiment 2.
- User identification via Sensor Group Model Ensemble (Section 5.6.3) and this will be illustrated in Experiment 3. 
- User identification in Open-World scenarios (Section 5.5.1) and this will be illustrated in Experiment 4.
- User identification in zero-day scenarios (Section 5.5.2) and this will be illustrated in Experiment 5.

### Experiments 

#### Experiment 1: User Identification with Sensor Groups
To evaluate the identification accuracy for all sensor groups and app adversary using the Random Forest model, with results from 20 apps and 20 users (refer to Section 5.1-5.4 in the main paper), run the following command:

```console
$ python run.py --adv='App' --Model='RF'
```
Alternatively, to use the XBoost Model, run:

```console
$ python run.py --adv='App' --Model='XGB'
```

#### Experiment 2: User Identification with Facial Emotion
To evaluate identification based on facial emotions with 20 users using the Random Forest model (refer to Sections 5.4.1 in the main paper), run the following command:

```console
$ python run.py --adv='emotion' --Model='RF'
```
Alternatively, to use the XBoost Model, run:

```console
$ python run.py --adv='emotion' --Model='XGB'
```

#### Experiment 3: User Identification with Multiple Sensor Group Ensemble
To ensemble multiple sensor group models using the Random Forest algorithm (refer to Section 5.5.3 in the main paper), run the following command: 

```console
$ python run.py --adv='Sensor_fusion' --Model='RF'
```
Alternatively, to use the XBoost Model, run:

```console
$ python run.py --adv='Sensor_fusion' --Model='XGB'
```

#### Experiment 4: User Identification in Open-World Scenario
To evaluate Open-world scenario with all sensor groups using the Random Forest algorithm (refer to Section 5.5.3 in the main paper), run the following command: 

```console
$ python run.py --adv='OW' --Model='RF'
```
Alternatively, to use the XBoost Model, run:

```console
$ python run.py --adv='OW' --Model='XGB'
```


#### Experiment 5: User Identification in Zero-Day Scenario
To evaluate Zero-day scenarios with 20 users and with all sensor groups using the Random Forest model (refer to Sections 4.2 and 5.5.2 in the main paper), run the following command:

```console
$ python run.py --adv='Zero-day' --Model='RF'
```
Alternatively, to use the XBoost Model, run:

```console
$ python run.py --adv='Zero-day' --Model='XGB'
```
