# Artifact Appendix

Paper title: **BehaVR:User Identification Based on VR Sensor Data**

Artifacts HotCRP Id: **#9** 

Requested Badge: **Available**

## Description

 **Overview:**
This repository contains BehaVR, a framework designed to analyze unique user identification using 20 commercial VR apps (See Section 2.3 in the main paper for details about selected apps) using VR sensor data (Body Motion or BM, Eye Gaze or EG, Hand Joint or HJ and Facial Expression or FE) collected by [Oculus Quest Pro](https://www.meta.com/quest/quest-pro/). BehaVR was developed for and utilized in the research paper BehaVR:User Identification Based on VR Sensor Data. Before using BehaVR, we recommend reading the paper for a comprehensive understanding of the framework. 

### Security/Privacy Issues and Ethical Concerns (All badges)
None.

## Basic Requirements (Only for Functional and Reproduced badges)
For detailed system requirements, please refer to the README files in their respective folders.

### Hardware Requirements
For detailed system requirements, please refer to the README files in their respective folders.

### Software Requirements
For detailed system requirements, please refer to the README files in their respective folders.

### Estimated Time and Storage Consumption
For detailed regarding estimated time and storage consumption, please refer to the README files in their respective folders.

## Environment 

### Accessibility (All badges)
- Codebase: https://github.com/UCI-Networking-Group/BehaVR/tree/main
- Associated Artifacts: 

### Set up the environment 
To clone the repository and navigate into the project folder, run the following commands:

```console
$ git clone https://github.com/UCI-Networking-Group/BehaVR.git
$ cd BehaVR
``` 
 BehaVR experiments consist of three parts: BehaVR Data collection, BehaVR Data processing, and BehaVR Adversaries.

- **BehaVR Data Collection**:
   - This module outlines the steps required to collect BehaVR data from the 20 apps discussed in the paper (see Section 3.1 in the paper).
   - To go to the `Data-collection` folder, use the following command: 
   
```console
$ cd Data-collection
``` 
   - Follow necessary steps in Data-collection README.md for environment set-up to collect data.
   
- **BehaVR Data Processing**:
   - This module outlines the steps to convert time series data into feature blocks for further feature engineering (see Section 4.1.1 and 4.1.2 in the paper).
   - To go to the `Data-processing` folder, use the following command: 
```console
$ cd Data-processing
``` 
   - Follow the necessary steps for environment setup in README.md to process time series data.

- **BehaVR Adversaries**:
   - This module outlines the necessary steps to design and evaluate BehaVR adversaries, including feature engineering and selection, model training, and evaluation (see Section 4.1.3-5 in the paper).
   - Go to the `Adversary` folder by using following command: 
       ```console
     $ cd Adversary
      ```
   - Follow the necessary steps in Adversary README.md for environment setup to train and evaluate BehaVR adversaries.

### Testing the Environment (Only for Functional and Reproduced badges)
For detailed regarding Testing the environment, please refer to the README files in their respective folders.


## Artifact Evaluation (Only for Functional and Reproduced badges)

### Main Results and Claims

#### Main Result 1: BehaVR Data Collection
In Section 3, we outline the BehaVR data collection process from real-world VR applications. Specifically:
- Utilizing the ALVR and SteamVR setup for data collection with the Oculus Quest Pro headset (Section 3.1)
- Conducting data collection from study participants (Section 3.2)

We will demonstrate how these processes were carried out in Experiment 1.

#### Main Result 2: Time Series Data Processing 
Sections 4.1.2 and 4.1.3 illustrate the processing of raw sensor data into feature blocks, transforming time series data to prepare for further feature engineering. 

We will demonstrate how these steps were implemented in Experiment 2.

#### Main Result 3: Identification results for BehaVR Adversaries
In Section 5, we detail user identification and related evaluations based on various adversarial settings. Specifically:

- User Identification with Sensor Groups, including Body Motion, Eye Gaze, Hand Joints, and Facial Expressions for the App adversary (Sections 5.1–5.4). This will be illustrated in Experiment 3.
- User identification via Facial Emotion Expression, discussed in Section 5.4.1 and this will be illustrated in Experiment 4.
- User identification in Open-World scenarios (Section 5.5.1) and this will be illustrated in Experiment 5.
- User identification in zero-day scenarios (Section 5.5.2) and this will be illustrated in Experiment 6.

#### Main Result 4: Feature Analysis
Top features for identification concerning app and device adversaries (Figure 8 and Table 10 in main paper) will be illustrated in Experiment 7.

### Experiments 

#### Experiment 1: BehaVR Data Collection
Once you start SteamVR, the ALVR server, and the (modified) ALVR client, you may use the following scripts to collect VR data.

`collect_data.sh` uses `adb logcat` to capture sensor readings and saves them into a file. Use it as follows:

```
$ collect_data.sh -f capture
```

`generate_all_csvs.sh` generates individual CSV files that record each type of sensor data:

```
$ generate_all_csvs.sh -f capture
```

This will generate a folder `capture/` containing the output CSV files.


#### Experiment 2: Time Series Data Processing
...

#### Experiment 3: User Identification with Sensor Groups
To evaluate the identification accuracy for all sensor groups and app adversary using the Random Forest model, with results from 20 apps and 20 users (refer to Section 5.1-5.4 in the main paper), run the following command:

```console
$ python run.py --adv='App' --Model='RF'
```
Alternatively, to use the XBoost Model, run:

```console
$ python run.py --adv='App' --Model='XGB'
```

#### Experiment 4:
To evaluate identification based on facial emotions with 20 users using the Random Forest model (refer to Sections 5.4.1 in the main paper), run the following command:

```console
$ python run.py --adv='emotion' --Model='RF'
```
Alternatively, to use the XBoost Model, run:

```console
$ python run.py --adv='emotion' --Model='XGB'
```

#### Experiment 6:
To evaluate Zero-day scenarios with 20 users and with all sensor groups using the Random Forest model (refer to Sections 4.2 and 5.5.2 in the main paper), run the following command:

```console
$ python run.py --adv='Zero-day' --Model='RF'
```
Alternatively, to use the XBoost Model, run:

```console
$ python run.py --adv='Zero-day' --Model='XGB'
```

## Limitations (Only for Functional and Reproduced badges)
Due to IRB restrictions, we are unable to share the original BehaVR dataset. However, we have provided all the necessary code to collect the BehaVR dataset, process the time series data, and train/evaluate BehaVR adversaries to reproduce all the tables and plots featured in the paper.

## Notes on Reusability (Only for Functional and Reproduced badges)
Please see the notes on Reusability to the README.md files in their respective folders.
