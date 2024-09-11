# BehaVR

**Abstract**
>Virtual reality (VR) platforms enable a wide range of applications; however, they pose unique privacy risks. In particular, VR devices are equipped with a rich set of sensors that collect personal and sensitive information (e.g., body motion, eye gaze, hand joints, and facial expressions). The data from these newly available sensors can be used to uniquely identify a user, even in the absence of explicit identifiers. In this paper, we seek to understand the extent to which a user can be identified based solely on VR sensor data, within and across real-world apps from diverse genres. We consider adversaries with capabilities that range from observing APIs available within a single app (app adversary) to observing all or selected sensor measurements across multiple apps on the VR device (device adversary).To that end, we introduce BehaVR, a framework for collecting and analyzing data from all sensor groups collected by multiple apps running on a VR device. We use BehaVR to collect data from real users interacting with 20 popular real-world apps. We use this data to build machine learning models for user identification both within and across apps, with features extracted from available sensor data. We show that these models can identify users with an accuracy of up to 100%, and we reveal the most important features and sensor groups, depending on the functionality of the app and the adversary. To the best of our knowledge, BehaVR is the first to comprehensively analyze user identification in VR, i.e., considering all sensor measurements available on consumer VR devices, collected by multiple real-world (as opposed to custom-made) apps.

**Overview**

This repository contains BehaVR, a framework designed to analyze unique user identification within VR apps using VR sensor data (Body Motion or BM, Eye Gaze or EG, Hand Joint or HJ and Facial Expression or FE) collected by [Oculus Quest Pro](https://www.meta.com/quest/quest-pro/). BehaVR was developed for and utilized in the research paper [BehaVR:User Identification Based on VR Sensor Data.](https://arxiv.org/pdf/2308.07304) Before using BehaVR, we recommend reading the paper for a comprehensive understanding of the framework. 

**Citation**

Please cite our paper as follows:

```bibtex
@inproceedings{,
  title     = {BehaVR: User Identification Based on VR Sensor Data},
  author    = {Jarin, Ismat and Duan, Yu and Trimananda, Rahmadi and Cui, Hao 
  and Elmalaki, Salma and Markopoulou, Athina},
  booktitle = {Proceedings on Privacy Enhancing Technologies (PoPETs)},
  volume    = {2025},
  issue     = {1},
  year      = {2025}
}
```

## Getting Started with BehaVR 

To clone the repository:

```console
$ git clone https://github.com/UCI-Networking-Group/BehaVR.git
$ cd BehaVR
``` 

**BehaVR Experiment Modules**

BehaVR experiments consist of three parts: BehaVR Data collection, BehaVR Data processing, and BehaVR Adversaries.

- **BehaVR Data Collection**:
   - We used ALVR to intercept and record sensor data in a SteamVR + Meta Quest Pro setup (see Section 3.1 in the paper).
   - Go to the `Data-collection` folder and follow the necessary steps for system requirements, environment setup and find relevent code in this folder.

- **BehaVR Data Processing**:
   - In the `Data-processing` folder, you'll find instructions on processing raw sensor data. This step converts time series data into feature blocks for further feature engineering.

- **BehaVR Adversaries**:
   - For feature engineering, model training, and evaluation tasks, refer to the `Adversary` folder for the appropriate scripts and instructions.

## System Requirements
For detailed system requirements, please refer to the README files in their respective folders.

## Environment Setup
For detailed environment setup, please refer to the README files in their respective folders.

## Usage
For detailed usage instructions for each module, please refer to the README files in their respective folders.
