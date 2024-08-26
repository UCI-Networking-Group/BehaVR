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

test1 = plt.figure()

num=range(1,21)
x_axis=['$a_1$','$a_2$','$a_3$','$a_4$','$a_5$','$a_6$','$a_7$','$a_8$','$a_9$','$a_{10}$','$a_{11}$','$a_{12}$','$a_{13}$','$a_{14}$','$a_{15}$','$a_{16}$','$a_{17}$','$a_{18}$','$a_{19}$','$a_{20}$']
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
