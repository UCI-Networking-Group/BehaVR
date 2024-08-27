#import necessary libraries
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns #grafikleştirme için

#plot identification accuracy comparisons for Body Motion (with and without headset features)
#load necessary results
acc=open('path/').readline().strip().split(',')
accController=open('path/').readline().strip().split(',')

#initialize plot
test1 = plt.figure()
num_apps = 20  # Set this to the number of labels you need
x_axis = [f'$a_{i+1}$' for i in range(num_labels)]
num=range(1,num_apps)

plt.plot(x_axis,accController,color='blue',marker='o',linestyle = 'None', label='--w/o Headset Features')
plt.plot(x_axis,acc,color='darkorange',marker='o',linestyle = 'None',label='--w Headset Features')
plt.vlines(x_axis, accController, acc, 'blue',linestyles ='--', alpha=.5,linewidth=.8)
plt.ylim([20,105])
plt.xlabel('App No.',fontsize=18)
plt.ylabel("Identification Accuracy",fontsize=18)
#plt.xticks(num, xticks)
plt.legend(loc=4,fontsize=16)

#save figure
directory='.../VR/BehaVR/results/graphs'
filename='AppAdv_Acc_motion20_cmp.pdf'
filepath = os.path.join(directory, filename)
test1.savefig(filepath)
