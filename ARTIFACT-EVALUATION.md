# Artifact Appendix

Paper title: **BehaVR:User Identification Based on VR Sensor Data**

Artifacts HotCRP Id: **#9** 

Requested Badge: **Available**

## Description

 **Overview:**
This repository contains BehaVR, a framework designed to analyze unique user identification using 20 commercial VR apps (See Section 2.3 in the main paper for details about selected apps) using VR sensor data (Body Motion or BM, Eye Gaze or EG, Hand Joint or HJ and Facial Expression or FE) collected by [Oculus Quest Pro](https://www.meta.com/quest/quest-pro/). BehaVR was developed for and utilized in the research paper BehaVR:User Identification Based on VR Sensor Data. Before using BehaVR, we recommend reading the paper for a comprehensive understanding of the framework. 

### Security/Privacy Issues and Ethical Concerns (All badges)
**Privacy Considerations**  
The data collection involves real-world participants, and the sensor data may include sensitive information or user fingerprints. We kindly ask reviewers to ensure that proper consent and precautions are followed. 

## Basic Requirements (Only for Functional and Reproduced badges)
While we are unable to share the BehaVR dataset at this time and are requesting only the available batch, we have included all the necessary information to verify functionality and reproducibility of BehaVR. Reviewers can refer to the respective README folders for detailed hardware and software requirements if they are interested.

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
   - Follow necessary steps in Data-collection README for environment set-up to collect data.
   
- **BehaVR Data Processing**:
   - This module outlines the steps to convert time series data into feature blocks for further feature engineering (see Section 4.1.1 and 4.1.2 in the paper).
   - To go to the `Data-processing` folder, use the following command: 
```console
$ cd Data-processing
``` 
   - Follow the necessary steps for environment setup in README to process time series data.

- **BehaVR Adversaries**:
   - This module outlines the necessary steps to design and evaluate BehaVR adversaries, including feature engineering and selection, model training, and evaluation (see Section 4.1.3-5 in the paper).
   - Go to the `Adversary` folder by using following command: 
       ```console
     $ cd Adversary
      ```
   - Follow the necessary steps in Adversary README for environment setup to train and evaluate BehaVR adversaries.

### Testing the Environment (Only for Functional and Reproduced badges)
For detailed regarding Testing the environment, please refer to the README files in their respective folders.


## Artifact Evaluation (Only for Functional and Reproduced badges)
While we are unable to share the BehaVR dataset at this time, reviewers who can collect and process their own dataset to reproduce the experiments can follow the steps outlined in the respective README folder.

### Main Results and Claims
For main results and claims, please refer to the README files in their respective folders.

### Experiments 
For experiments, please refer to the README files in their respective folders.

## Limitations (Only for Functional and Reproduced badges)
Due to IRB restrictions, we are unable to share the original BehaVR dataset. However, we have provided all the necessary code to collect the BehaVR dataset, process the time series data, and train/evaluate BehaVR adversaries to reproduce all the tables and plots featured in the paper.

## Notes on Reusability (Only for Functional and Reproduced badges)
Please see the notes on Reusability to the README files in their respective folders.
