#import necessary libraries
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns #grafikleştirme için

#initialization
clm=['Social','Archery','Flight Sim.','Golfing','Int.Nav.','Knu.Walk.','Rhythm','Shooting','Teleport.'] #column name for app group
clr=['r','lime','b','y','orange','mediumslateblue','c','g','m'] #Different color for different app group
num_apps = 20  # Set this to the number of labels you need
x_axis = [f'$a_{i+1}$' for i in range(num_labels)]
num=range(1,num_apps)

#plot identification accuracy comparisons for Body Motion (with and without headset features)
#load necessary results
acc=open('path/').readline().strip().split(',')
accController=open('path/').readline().strip().split(',')

#initialize plot
test1 = plt.figure()


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


#comparison between FBL and FBA
accFTN=open('path/').readline().strip().split(',')
test2 = plt.figure()
plt.plot(x_axis,accFTN,color='blue',marker='o',linestyle = 'None', label='FBL')
plt.plot(x_axis,accBar,color='red',marker='o',linestyle = 'None',label='FBA')
plt.vlines(x_axis, accFTN, accBar, 'blue',linestyles ='--', alpha=.5,linewidth=.8)
plt.ylim([50,105])
plt.xlabel('App No.',fontsize=18)
plt.ylabel("Identification Accuracy",fontsize=18)
#xticks = ['data5B','data2B','data1B','data.5B']
#plt.xticks(num, xticks)
plt.legend(loc=4,fontsize=15)


#save figure
directory='.../VR/BehaVR/results/graphs'
filename='AppAdv_Acc_motion20_cmp_block.pdf'
filepath = os.path.join(directory, filename)
test2.savefig(filepath)

#plot minimum time
average_time=open('path/').readline().strip().split(',') #average time for each app group
acc_group=open('path/').readline().strip().split(',') #identification accuracy per app group (each sample app from each group) per average time

open('path/').readline().strip().split(',')
test3 = plt.figure()
markers=['o']
for i in range(len(g)):
  plt.plot(average_time,acc_group[i],color=clr[i],marker='o',linestyle = '--',linewidth=.5, label=r'$a_{'+str(g[i])+'}$'+'('+clm[i]+')')

plt.xlabel("Sub-session time ($S_t$) Per User",fontsize=18)
plt.ylabel("Identification Accuracy",fontsize=18)
plt.ylim([25,105])
plt.xlim([0,100])

test3.show()
plt.legend(ncol=2,loc='lower right',fontsize=10)

#save figure
directory='.../VR/BehaVR/results/graphs'
filename='AppAdv_AccBlockPerUser_Motion.pdf'
filepath = os.path.join(directory, filename)
test3.savefig(filepath)

