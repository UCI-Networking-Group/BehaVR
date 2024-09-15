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

### Set up the environment (Only for Functional and Reproduced badges)
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
   - Follow the necessary steps for environment setup to process time series data.

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
This section includes all the steps required to evaluate your artifact's functionality and validate your paper's key results and claims.
Therefore, highlight your paper's main results and claims in the first subsection. And describe the experiments that support your claims in the subsection after that.

### Main Results and Claims
List all your paper's results and claims that are supported by your submitted artifacts.

#### Main Result 1: Name
Describe the results in 1 to 3 sentences.
Refer to the related sections in your paper and reference the experiments that support this result/claim.

#### Main Result 2: Name
...

### Experiments 
List each experiment the reviewer has to execute. Describe:
 - How to execute it in detailed steps.
 - What the expected result is.
 - How long it takes and how much space it consumes on disk. (approximately)
 - Which claim and results does it support, and how.

#### Experiment 1: Name
Provide a short explanation of the experiment and expected results.
Describe thoroughly the steps to perform the experiment and to collect and organize the results as expected from your paper.
Use code segments to support the reviewers, e.g.,
```bash
python experiment_1.py
```
#### Experiment 2: Name
...

#### Experiment 3: Name 
...

## Limitations (Only for Functional and Reproduced badges)
Due to IRB restrictions, we are unable to share the original BehaVR dataset. However, we have provided all the necessary code to collect the BehaVR dataset, process the time series data, and train/evaluate BehaVR adversaries to reproduce all the tables and plots featured in the paper.

## Notes on Reusability (Only for Functional and Reproduced badges)
First, this section might not apply to your artifacts.
Use it to share information on how your artifact can be used beyond your research paper, e.g., as a general framework.
The overall goal of artifact evaluation is not only to reproduce and verify your research but also to help other researchers to re-use and improve on your artifacts.
Please describe how your artifacts can be adapted to other settings, e.g., more input dimensions, other datasets, and other behavior, through replacing individual modules and functionality or running more iterations of a specific part.
