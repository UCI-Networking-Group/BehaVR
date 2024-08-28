# README.md
##Content
This section demonstrates how the raw sensor data has been processed to convert time series data into feature blocks, laying the groundwork for subsequent feature engineering. 

## Dependencies
    - Anaconda3 (version 1.7.2) 
    - Jupyter Notebook (version 6.1.7)
    - Python (version 3.7.4)
    - numpy (version 1.21.6)
    - pandas (version 1.2.3)
    - matplotlib (version 3.3.4)



## Code 
Below are listed in folder/file id. 
1. REG: Reorganize the data files, from folder per user per game per round to per sensor group.
2. PROCESSING: Process four different sensor group data using two different block division methods.
    1. Four different sensor groups include body motion(SG1), eye gaze(SG2), hand joints(SG3), facial expression(SG4).
    2. Two different block division methods include FBA(or called FBN, means fixed block amount), FBL(or called FTN, means fixed block length).
3. PLOT: Generate sensor data duration overview, Fig.3 (a) & (b).
