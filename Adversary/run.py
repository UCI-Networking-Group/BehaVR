#@ijarin
import os

# Relative path to the main.py script
script_path = '..../Adversary/code/main.py'

#to run an app adversary model on Facial Expression or FE sensor group on first 15 apps, where your chosen model is Random Forest, run the following command on the terminal:
os.system(f"python {script_path} --SG='BM'--num_app=15 --adv='App' --Model='RF'")

 '''
#If we want to run the command on App adversary under BehaVR assumptions on all four sensor data
arrSG=['BM','FE','EG','HJ']
for sg in arrSG:
    os.system(f"python {script_path} --SG={sg} --Model='RF' --adv='App'")
'''
