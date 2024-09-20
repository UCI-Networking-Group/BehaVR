# @ijarin
# script to run BehaVR
import os
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description='Run BehaVR with specified adversary and model.')
parser.add_argument('--adv', type=str, required=True, help='Specify the adversary')
parser.add_argument('--model', type=str, required=True, help='Specify the model, RF or XGB')

args = parser.parse_args()

# Relative path to the main.py script
script_path = '../Adversary/code/main.py'

# App adversary model with all sensor groups
if args.adv == 'App':
    SG = ['BM', 'FE', 'EG', 'HJ']
    for sg in SG:
        os.system(f"python {script_path} --SG={sg} --Model='{args.model}' --adv='{args.adv}'")


# Identification with Facial Emotion Expression with FE sensor group
if args.adv == 'emotion':
    SG = ['FE']
    for sg in SG:
        os.system(f"python {script_path} --SG={sg} --Model='{args.model}' --adv='{args.adv}'")

# Identification with ensemble multiple sensor group models
if args.adv == 'Sensor_fusion':
    SG1 = ['BM','HJ']
    for sg in SG1:
        os.system(f"python {script_path} --SG1={sg} --SG2='EG' --Model='{args.model}' --adv='{args.adv}'")

# Open-world Scenario with all sensor groups
if args.adv == 'OW':
    SG = ['BM', 'FE', 'EG', 'HJ']
    for sg in SG:
        os.system(f"python {script_path} --SG={sg} --Model='{args.model}' --adv='{args.adv}'")
        

# App adversary model with all sensor groups
if args.adv == 'Zero-day':
    SG = ['BM', 'FE', 'EG', 'HJ']
    for sg in SG:
        os.system(f"python {script_path} --SG={sg} --Model='{args.model}' --adv='{args.adv}'")
