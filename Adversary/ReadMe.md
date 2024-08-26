##Summary of the paper: 
VR platforms offer diverse applications but raise privacy concerns due to their sensors, which can uniquely identify users. This study introduces BehaVR, a framework for collecting and analyzing sensor data from multiple VR apps to assess user identification risks. Using real user data from 20 apps, the research builds machine learning models that achieve up to 100% accuracy in identifying users based on sensor data. BehaVR is the first comprehensive analysis of user identification in VR across real-world apps, highlighting key features and sensor groups based on app functionality and adversary capabilities.

This folder includes code for feature engineering, design and evaluating BehaVR adversary, analyze top features etc. 

##Dependencies: 
Please see requirement.txt file for all the dependencies.

##Running Code: 
   \Adversary:
       run.py`: Automatically runs the main file and please adjusts for different adversarial scenarios, like app adversary, open-world, or zero-day scenarios. 
       \code:
           Input_data.py: Contain functions for load and preprocess input data from different sensor Group, BM=Body motion, EG= Eye Gaze, HJ=Hand Joint and FE=Facial Expression
           model.py: Contain necessary functions for tuning model, loading necessary models depending on the sensor Group and finding top features
           preprocess.py: Contain necessary functions for data preprocessing, feature engineering, loading train/test data and adversarial conditions
           main.py: Main file generate output results depending on the type of adversaries. main.py contain all the variable parameters to run the model in different settings.
           sample_plot.py: Plot some sample graphs that can be found on the paper. We used saved results to plot those graph 
      

To run the code, please run the commands on your terminal based on the type of BehaVR adversary and sensor group. For example, to run an app adversary model on Facial Expression or FE sensor group on first 15 apps, where your chosen model is Random Forest, run the following command on the terminal:
                $cd code
                $python main.py --SG='BM'--num_app=15 --adv='App' --Model='RF' 
or 
You can just simply change your command in the run.py and simply run the following command in the terminal: 
                $python run.py 
     


