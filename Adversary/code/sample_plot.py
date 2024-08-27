#@ijarin
#example plot
import matplotlib.pyplot as plt
import seaborn as sns #grafikleştirme için
import matplotlib.pyplot as plt
import os


#Plot the effect of important features on identification accuracy
acc_allFeatures=[ 100.,  95.,  95.,  70.,  95.,  80.,  95.,  80.,  85.,  95.,  95.,
        85.,  90.,  100.,  95.,  95., 100.,  95.,  100.,  95.] #Identification accuracy for BM with all features

acc_wo_HeadsetFeatures=[80., 80., 80., 65., 60., 60., 60., 55., 50., 75., 80., 55., 60.,
       95., 90., 75., 80., 60., 85., 85.] #Identification accuracy for BM without headset_features



#initialize the plot
test1 = plt.figure()


# Set this to the number of apps you need
num_apps = 20 #number of total apps

#Define x_axis for each app
x_axis = [f'$a_{i+1}$' for i in range(num_labels)]
num=range(1,num_apps)


#plot
plt.plot(x_axis,acc_wo_HeadsetFeatures,color='blue',marker='o',linestyle = 'None', label='--w/o Headset Features')
plt.plot(x_axis,acc_allFeatures,color='darkorange',marker='o',linestyle = 'None',label='--w Headset Features')
plt.vlines(x_axis, acc_wo_HeadsetFeatures, acc_allFeatures, 'blue',linestyles ='--', alpha=.5,linewidth=.8)
plt.ylim([20,105])
plt.xlabel('App No.',fontsize=18)
plt.ylabel("Identification Accuracy",fontsize=18)
plt.legend(loc=4,fontsize=16)


#save figure
directory='.../VR/BehaVR/results/graphs'
filename='AppAdv_Acc_motion20_cmp.pdf'
filepath = os.path.join(directory, filename)
test1.savefig(filepath)
