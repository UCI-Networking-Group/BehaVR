## Summary of the paper
VR platforms offer diverse applications but raise privacy concerns due to their sensors, which can uniquely identify users. This study introduces BehaVR, a framework for collecting and analyzing sensor data from multiple VR apps to assess user identification risks. Using real user data from 20 apps, the research builds machine learning models that achieve up to 100% accuracy in identifying users based on sensor data. BehaVR is the first comprehensive analysis of user identification in VR across real-world apps, highlighting key features and sensor groups based on app functionality and adversary capabilities.

This folder includes code for feature engineering, design and evaluating BehaVR adversary, analyzing top features, etc.

## Dependencies
Please see the `requirement.txt` file for all the dependencies.

## Running Code
- **Adversary:** `run.py`: Automatically runs the main file and adjusts for different adversarial scenarios, like app adversary, open-world, or zero-day scenarios.
- **Code:**
  - `Input_data.py`: Contains functions for loading and preprocessing input data from different sensor groups: BM (Body Motion), EG (Eye Gaze), HJ (Hand Joint), and FE (Facial Expression).
  - `model.py`: Contains necessary functions for tuning models, loading necessary models depending on the sensor group, and finding top features.
  - `preprocess.py`: Contains necessary functions for data preprocessing, feature engineering, loading train/test data, and adversarial conditions.
  - `main.py`: Main file that generates output results depending on the type of adversaries. `main.py` contains all the variable parameters to run the model in different settings.
  - `sample_plot.py`: Plots some sample graphs that can be found in the paper. We used saved results to plot those graphs.

To run the code, please run the commands on your terminal based on the type of BehaVR adversary and sensor group. For example, to run an app adversary model on the Facial Expression (FE) sensor group on the first 15 apps, where your chosen model is Random Forest, run the following command in the terminal:

```bash
$ cd code
$ python main.py --SG='BM' --num_app=15 --adv='App' --Model='RF'
Or, you can simply change your command in the run.py and run the following command in the terminal:
$ python run.py

