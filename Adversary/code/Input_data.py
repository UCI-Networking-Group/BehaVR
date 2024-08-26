#This file contain the functions to call different VR sensor data, perform initial preprocessing and return data 
#@ijarin
import numpy as np
import pandas as pd
from preprocess import change_nameFace, change_nameF, change_nameH, change_nameEye,change_nameE, RLeft_Right, change_nameHand, change_eliminate_Head, change_eliminate_RightHand, Emotion_units, change_nameFacialEexpression


def Final_feature(SG,OW):
    if SG=='BM': #Body Motion Data
        if OW==False:
            #call abstracted Body Motion data for 20 users
            data2 =pd.read_csv('.../VR/data/BM/SG1_FBN_r2_feature.csv', sep=',')
            data1 = pd.read_csv('.../VR/data/BM/SG1_FBN_r1_feature.csv', sep=',')
            data_05 = pd.read_csv('.../VR/data/BM/SG1_FBN_r0.5_feature.csv', sep=',')
        else:
            print("Open World Data as Input:")
            #call abstracted Body Motion for open world settings
            data2 =pd.read_csv('/data/...', sep=',')
            data1 = pd.read_csv('/data/...', sep=',')
            data_05 = pd.read_csv('/data/...', sep=',')
            

        #preprocess the data
        datar1=change_nameH(data1)
        datar2=change_nameH(data2)
        datar_05=change_nameH(data_05)
        D=[datar2,datar1,datar_05]

    elif SG=='EG': #Eye Tracking/Gaze Data
        if OW==False:
            #call abstracted Eye Gaze data for 20 users
            data2 =pd.read_csv('/data/..', sep=',')
            data1 = pd.read_csv('/data/..', sep=',')
            data_05 = pd.read_csv('/data/..', sep=',')
        else:
            print("Open World Data as Input:")
            data2 =pd.read_csv('/data/..', sep=',') #collect openworld data
            data1 = pd.read_csv('/data/..', sep=',')
            data_05 = pd.read_csv('/data/..', sep=',')
        
        datar1=change_nameE(data1)
        datar2=change_nameE(data2)
        datar_05=change_nameE(data_05)
        
        #preprocess the data
        dataE=[datar2,datar1,datar_05]
        
        dataEye=[]
        for i in range(len(dataE)):
          d=dataE[i]
          dnew1=RLeft_Right(d)
          dnew=change_nameEye(dnew1)
          dataEye.append(dnew)
        #Final eyedata
        D=dataEye


    elif SG=='HJ':  #Hand joint or Hand Tracking data
        if OW==False:
            #call abstracted Hand Joint data for 20 users
            data2 =pd.read_csv('/data/..', sep=',')
            data1 = pd.read_csv('/data/..', sep=',')
            data_05 = pd.read_csv('/data/..', sep=',')
        else:
            print("Open World Data as Input:")
            data2 =pd.read_csv('/data/..', sep=',') #collect openworld data
            data1 = pd.read_csv('/data/..', sep=',')
            data_05 = pd.read_csv('/data/..', sep=',')
        
        datar1=change_nameHand(data1)
        datar2=change_nameHand(data2)
        datar_05=change_nameHand(data_05)
        
        #preprocess the data
        D=[datar2,datar1,datar_05]
        
    
    elif SG=='FE':
        if OW==False:
            #call abstracted Facial data for 20 users
            data2 =pd.read_csv('.../data/FE/SG4_FBN_r2_feature.csv', sep=',')
            data1 = pd.read_csv('.../VR/data/FE/SG4_FBN_r1_feature.csv', sep=',')
            data_05 = pd.read_csv('.../VR/data/FE/SG4_FBN_r0.5_feature.csv', sep=',')
        else:
            #call abstracted Body Motion for open world settings
            print("Open World Data as Input:")
            data2 =pd.read_csv('/data/...', sep=',')
            data1 = pd.read_csv('/data/...', sep=',')
            data_05 = pd.read_csv('/data/...', sep=',')
        
        #Preprocess the data
        datar2=change_nameF(data2)
        datar1=change_nameF(data1)
        datar_05=change_nameF(data_05)
        D=[datar2,datar1,datar_05]
        
    return D


def Feature_elimination(SG):
    if SG=='BM': #Body Motion Data
        #call abstracted Body Motion data for 20 users
        data2 =pd.read_csv('.../data/BM/SG1_FBN_r2_feature.csv', sep=',')
        data1 = pd.read_csv('...data/BM/SG1_FBN_r1_feature.csv', sep=',')
        data_05 = pd.read_csv('.../BM/SG1_FBN_r0.5_feature.csv', sep=',')
        
        #Filter the data such that headset features are going to be eliminated
        datar1=change_eliminate_Head(data1)
        datar2=change_eliminate_Head(data2)
        datar_05=change_eliminate_Head(data_05)
        
        DE=[datar2,datar1,datar_05]
    
    elif SG=='EG': #Eye Tracking/Gaze Data
        #call abstracted Eye Gaze data for 20 users
        data2 =pd.read_csv('/data/..', sep=',')
        data1 = pd.read_csv('/data/..', sep=',')
        data_05 = pd.read_csv('/data/..', sep=',')

        #data without augmented features
        datar1=change_nameEye(change_nameE(data1))
        datar2=change_nameEye(change_nameE(data2))
        datar_05=change_nameEye(change_nameE(data_05))
        
        #final
        DE=[datar2,datar1,datar_05]


    elif SG=='HJ': #Hand joint data
        #call abstracted Hand Joint data for 20 users
        data2 =pd.read_csv('/data/..', sep=',')
        data1 = pd.read_csv('/data/..', sep=',')
        data_05 = pd.read_csv('/data/..', sep=',')

        #data without Right Hand Features
        datar1=change_eliminate_RightHand(change_nameHand(data1))
        datar2=change_eliminate_RightHand(change_nameHand(data2))
        datar_05=change_eliminate_RightHand(change_nameHand(data_05))
        
        #final
        DE=[datar2,datar1,datar_05]
    
    elif SG=='FE': #Eye Tracking/Gaze Data
        #call abstracted Hand Joint data for 20 users
        data2 =pd.read_csv('/data/..', sep=',')
        data1 = pd.read_csv('/data/..', sep=',')
        data_05 = pd.read_csv('/data/..', sep=',')
        _,f=Emotion_units()

        #data without Facial Emotion Features
        datar1=change_nameFacialEexpression(data1,f)
        datar2=change_nameFacialEexpression(data2,f)
        datar_05=change_nameFacialEexpression(data_05,f)
        
        #final
        DE=[datar2,datar1,datar_05]
    
    return DE
    
        

