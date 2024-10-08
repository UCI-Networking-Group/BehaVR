#@ijarin
#Run to train and evaluate different BehaVR adversaries
#initialize the required libraries
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import argparse
from Input_data import Final_feature, Feature_elimination
from preprocess import data_preProcess, dT,train_test,stratified_train_test_split, index_dev,divide_pred, final_label, Emotion, DataE, Emotion_units, concatenate_arrays, app_groups_name, app_grouping, f_data
from model import RF_tuning, XB_tuning, final_model, Top_Features
import os


#arguments
parser = argparse.ArgumentParser()

#Which sensor data/sensor group we are analyzing, available arguments: 'BM' for Body Motion, 'EG' for Eye Tracking, 'HJ' for Hand Joints and 'FE' for Facial Expression
parser.add_argument('--SG', type=str, help='The sensor group-BM/FE/EG/HJ',default='BM')

#types of adversarys, available arguments: adv='App' for app adversary, 'emotion' for adversary who only consider emotion features, 'OW' for open-world settings, 'Zero-Day' for zero day settings
#'Sensor_fusion' for combining multiple sensors
parser.add_argument('--adv', type=str, help='Type of Adversary- App:App Adversary, emotion:Identification using facial emotion, Sensor_fusion:Sensor Group Model Ensemble, OW:Open World Setting, Zero-Day:Zero-Day Settings.',default='App')


#Number of emotion we analyzing
parser.add_argument('--n_emo', type=int, help='The number of emotions we consider',default=2)

#if use Sensor fusion as adversary, use SG1, SG2
parser.add_argument('--SG1', type=str, help='The first sensor group for Ensemble multiple sensors-BM/FE/EG/HJ',default='BM')
parser.add_argument('--SG2', type=str, help='The second sensor group for Ensemble multiple sensors-BM/FE/EG/HJ',default='EG')

# Set True if Open World Setting is True, otherwise, False
parser.add_argument('--OW', type=bool, help='If open-world setting is true',default=False)

#add arguments to choose FBA or FTN method
parser.add_argument('--data_process', type=str, help='FBA or FBL',default='FBA')

#If feature elimination setting (evaluate by eliminating certain types of features, example Headset features for BM) is true, set True, otherwise False
parser.add_argument('--feature_elim', type=bool, help='Eliminating top features',default=False)

#total number of users/participants
parser.add_argument('--num_user', type=int, help='Total number of users',default=20)

#if remove any user
parser.add_argument('--rt', type=str, help='remove user id x ir rt=t, all users will be used if rt=f',default='f')

#Total number of apps we are analyzing, in BehaVR it's 20
parser.add_argument('--num_app', type=int, help='Total number of apps',default=20)

#Set up which type of model architecture adversary choosing
parser.add_argument('--Model', type=str, help='Model type, RF=Random Forest, XGB=Xboost',default='RF')

#target classifier, for BehaVR we want to classify each user uniquely
parser.add_argument('--target', type=str, help='Target Classifier',default='user_id')

#number of Cross_validation for model training
parser.add_argument('--cross_val', type=int, help='Cross validation value',default=5)

#control block length ratio
parser.add_argument('--ratio', type=int, help='Block length controller ratio',default=1)

#controlling block length ratio
parser.add_argument('--r1', type=int, help='block length controller ratio for SG1',default=1)
parser.add_argument('--r2', type=int, help='block length controller ratio for SG2',default=1)

#control subsession time, example, M=3 means 33.33% subsession time will be considered for evaluation
parser.add_argument('--M', type=int, help='Controlling sub-session time, M inversely proportional to subsession',default=1)

#observe f_n number of top features
parser.add_argument('--f_n', type=int, help='How many top feature we want to observe',default=20)

#put output directorty to save the results
parser.add_argument('--output_dir', type=str, help='Directory to save output results',default='.../VR/Adversary/results/output')


args = parser.parse_args()

#initialization data, parameters
#initialize game ids
g_id=list(range(1, args.num_app + 1)) #game_id represents 1 to 20 app for our experiments; for example, g_id=1 is BeatSaber
#user_initialization
num_user=args.num_user
sk = list(range(1, num_user + 1))

#block length ratios
rp=[args.ratio] #choose block length, 0=0.5, 1=1, 2=2

# Initialize an array to store the ratio values
if (len(rp)==1):
    rp_arr=rp
    print("chosen block ratio is:",rp_arr)
else:
    rp_arr=list(range(len(rp)+1))
    print("chosen block ratio is:",rp_arr)

# Initialize an array to hold the index values of maximum accuracy
index_rp=np.zeros(len(g_id))

#block length ratios for sensor fusion
rp1=args.r1
rp2=args.r2

#Boolean parameter to determine if all users are included in the consideration
rt=args.rt

# Remove specific users from the existing user pool
if (rt=='t'):
    for p in Non_users:
        sk.remove(p)

# Argument to specify the type of adversary to be used
adv=args.adv

#choose data processing method (FBN/FTN)
data_process=args.data_process

# Define the sensor group to be used
SG=args.SG

#define the sensor group for sensor fusion
SG1=args.SG1 #Define 1st sensor group if perform sensor fusion
SG2=args.SG2 #Define 2nd sensor group if pereform sensor fusion

#controlling subsession length
M=args.M

# Specify the number of emotional states being considered
n_emo=args.n_emo

#number of top features
f_n=args.f_n

#number of cross validation in model training
cross_val=args.cross_val

#target class
target=args.target

#Open-World Settings
OW=args.OW

#feature_elim enabled/disabled
feature_elim=args.feature_elim

#model initialization
Model=args.Model #Type of models

# Initialize app grouping settings
gname=app_groups_name() #call app grpups
acc=np.zeros((len(g_id),len(rp_arr))) # attack accuracy for different ratio
accGroup=np.zeros((len(gname),len(gname)))

# Array parameters for storing identification accuracy values in different stages
final_labels=np.zeros((len(g_id),num_user))
accBar=np.zeros(len(g_id)) #attack accuracy
models=[] #save each models for each app
#save final accuracy
accfinal=[]
#Accuracy for corresponding emotions in each app
acc_emotion=np.zeros((len(g_id),n_emo))

#output directory
output_dir = args.output_dir
#save final features
final_features=[]


#calculate the optimizationratio
# Uses the default block ratio length since optimization is not applied
#calculate optimization ratio (for block length) from app adversaries

#if rp_arr contains only one fixed value, block ratio is fixed
if len(rp_arr)==1:
    opt_rp=rp[0]

#if rp_arr contains multiple values, find the optimized block ratio that provides the highest identification accuracy
else:
    #call input sensor data
    data=Final_feature(SG,OW,data_process) #call input processed data (containing final features per block)
    
    # Find the optimized ratio for each app by iterating over each app
    for j in range(len(g_id)):
    
    #iterate over all block ratio
      for i in range(len(rp_arr)):
         rp_i=rp_arr[i]
         d=data[rp_i]
         d_h=data_preProcess(d,g_id[j],args.target)
         print("check unique users id:",np.unique(d_h['user_id'].values))
         print("load train/test dataset")
         X_train,y_train,X_test,y_test,X_val,y_val=train_test(d_h,M,rt,target,sk)
         print('size of training data',X_train.shape)
         sd=index_dev(y_test) #find the 'sub session' division point
         #select Model and change data according to model type
         if Model=='RF':
            print('Chosen model is Random Forest')
         elif Model=='XGB':
            print('Chosen model is Xboost')
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test=le.fit_transform(y_test)
         #call final model
         model= final_model(Model,SG,cross_val, X_train,y_train)

         models.append(model) #save each model for each app
         y_pred = model.predict(np.array(X_test)) #prediction for each block
         new_pred=np.array(divide_pred(y_pred,sd),dtype=object) #divided prediction
         true_preds=np.array(divide_pred(y_test,sd),dtype=object)
         true_labels=final_label(true_preds,len(sd))
         #true_labels=final_label(np.array(divide_pred(y_test,sd),dtype=object).astype(int),len(sd))
         final_labels[j,:]=final_label(new_pred,len(sd)) #calculate final label based on 'sub session'
         acc[j,i]=accuracy_score(true_labels, final_labels[j,:])*100 #accuracy per app per ratio
      
      #collect top features
      feature=Top_Features(model,g_id[j],f_n,X_train)
      final_features.append(feature)
      
      #final identification accuracy (highest over all block ratio)
      accBar[j]=np.max(acc[j,:])
      
      #index of higest accuracy, same as block ratio for higest accuracy
      index_rp[j]=np.argmax(acc[j, :])
    
    #store the optimized ratio (block length) that provides best identification accuracy
    print("final identification accuracy for app a_"+str(j+1)+" is "+str(accBar[j])+"\%")
    values, counts = np.unique(index_rp, return_counts=True)
    opt_rp = values[np.argmax(counts)]
    print("Optimized block length ratio:", opt_rp)



#data=Final_feature(SG) #Call the input proccessed data

#Adversarial settings=App adversary
if (adv=='App'):
    if feature_elim==True: #if we eliminate important features and see identification accuracy
        data=Feature_elimination(SG)
    else: #use all features obtained after processing & feature Engineering
        data=Final_feature(SG,OW,data_process) #call input processed data (containing final features per block)
    for j in range(len(g_id)):
    # Selects the optimal block ratio length based on the input data
      d=data[opt_rp]
      d_h=data_preProcess(d,g_id[j],args.target)
      print("check unique users id:",np.unique(d_h['user_id'].values))
      print("load train/test dataset")
      X_train,y_train,X_test,y_test,X_val,y_val=train_test(d_h,M,rt,target,sk)
      print('size of training data',X_train.shape)
      sd=index_dev(y_test) #find the 'sub session' division point
      #select Model and change data according to model type
      if Model=='RF':
        print('Chosen model is Random Forest')
      elif Model=='XGB':
        print('Chosen model is Xboost')
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test=le.fit_transform(y_test)
         #call final model
      model= final_model(Model,SG,cross_val, X_train,y_train)

      models.append(model) #save each model for each app
      
      #calculate prediction
      y_pred = model.predict(np.array(X_test)) #prediction for each block
      new_pred=np.array(divide_pred(y_pred,sd),dtype=object) #divided prediction
      true_preds=np.array(divide_pred(y_test,sd),dtype=object)
      true_labels=final_label(true_preds,len(sd))

      #calculate final labels
      final_labels[j,:]=final_label(new_pred,len(sd)) #calculate final label based on 'sub session'
      
      #store identification accuracy for each app
      accBar[j]=accuracy_score(true_labels, final_labels[j,:])*100 #accuracy per app per ratio
      
      #collect top features
      feature=Top_Features(model,g_id[j],f_n,X_train)
      final_features.append(feature)
      
      #print and store identification accuracy for each of all apps
      print("final identification accuracy for app a_"+str(j+1)+" is "+str(accBar[j])+"\%")
      app_no='a_'+str(g_id[j])
      accuracy=str(accBar[j])+'%'
      accfinal.append((app_no, accuracy))

#Adversarial settings=Facial Emotions
elif (adv=='emotion'): #adversary consider particular emotions for identification
    data=Final_feature(SG,OW,data_process) #call input processed data (containing final features per block)
    for j in range(len(g_id)):
      for i in range(n_emo):
         emo,_=Emotion_units()
         fE=emo[i]
         d1=data[opt_rp]
         d=Emotion(d1,fE) #consider features w.r.t an emotion
         d_h=data_preProcess(d,g_id[j],args.target)
         print("check unique users id:",np.unique(d_h['user_id'].values))
         print("load train/test dataset")
         X_train,y_train,X_test,y_test,X_val,y_val=train_test(d_h,M,rt,target,sk)
         print('size of training data',X_train.shape)
         sd=index_dev(y_test) #find the 'sub session' division point
         #select Model
         if Model=='RF':
            print('Chosen model is Random Forest')
         elif Model=='XGB':
            print('Chosen model is Xboost')
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test=le.fit_transform(y_test)
         
         model=final_model(Model,SG,cross_val, X_train,y_train)
         models.append(model) #save each model for each app
         y_pred = model.predict(np.array(X_test)) #prediction for each block
         new_pred=np.array(divide_pred(y_pred,sd),dtype=object) #divided prediction
         true_preds=np.array(divide_pred(y_test,sd),dtype=object)
         true_labels=final_label(true_preds,len(sd))
         #true_labels=final_label(np.array(divide_pred(y_test,sd),dtype=object).astype(int),len(sd))
         final=final_label(new_pred,len(sd)) #calculate final label based on 'sub session'
         acc_emotion[j,i]=accuracy_score(true_labels, final)*100 #accuracy per app per ratio
      print("final identification accuracy for app a_"+str(j+1)+" is "+str(acc_emotion[j,:])+"\%")
      app_no='a_'+str(g_id[j])
      accfinal.append((app_no,acc_emotion[j,:]))


#Adversarial settings=Multiple Sensor Fusion
elif (adv=='Sensor_fusion'): #adversary consider fusing two or more sensors if one of the sensors identification accuracy is not good enough; optimization for param r is not required as adv already optimized for single sensors.
    data1=Final_feature(SG1,OW,data_process)
    data2=Final_feature(SG2,OW,data_process)
    for j in range(len(g_id)):
        d=data1[rp1] #ratio of FBA=1
        d1=data2[rp2]
        #preprocess data
        d_h=data_preProcess(d,g_id[j],target)
        d_h1=data_preProcess(d1,g_id[j],target)
        
        #Train-Test division for both dataset
        X_train,y_train,X_test,y_test,X_val,y_val=train_test(d_h,M,rt,target,sk)
        X_train1,y_train1,X_test1,y_test1,X_val1,y_val1=train_test(d_h1,M,rt,target,sk)
        sd=index_dev(y_test) #find the 'sub session' division point
        sd1=index_dev(y_test1)
        print(f"Size of training data for {SG1} and {SG2} are", X_train.shape, X_train1.shape)

        #select Model for data1 and data2, adversary can choose different model as well
        if Model=='RF':
            print('Chosen model is Random Forest')
        elif Model=='XGB':
            print('Chosen model is Xboost')
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test=le.fit_transform(y_test)
        
        model=final_model(Model,SG1,cross_val, X_train,y_train)
        model1=final_model(Model,SG2,cross_val, X_train1,y_train1)
        
        #prediction
        y_pred = model.predict(np.array(X_test))
        y_pred1 = model1.predict(np.array(X_test1))
        
        #new prediction based on per user per sensor
        new_pred=np.array(divide_pred(y_pred,sd),dtype=object)
        new_pred1=np.array(divide_pred(y_pred1,sd1),dtype=object)
        
        #fuse prediction for selected sensors
        new_preds = concatenate_arrays(new_pred, new_pred1)
        
        #actual prediction/true prediction label per user
        true_pred=np.array(divide_pred(y_test,sd),dtype=object)
        true_pred1=np.array(divide_pred(y_test1,sd1),dtype=object)
        true_preds = concatenate_arrays(true_pred, true_pred1)
        #print('true preds',true_preds)
        final_labels[j,:]=final_label(new_preds,len(sd+sd1)) #calculate final label based on 'sub session'
        true_labels=final_label(true_preds,len(sd+sd1))
        accBar[j]=accuracy_score(true_labels, final_labels[j,:])*100
        
        #collect top features
        feature=Top_Features(model,g_id[j],f_n,X_train)
        final_features.append(feature)
        
        print("final identification accuracy for app a_"+str(j+1)+" is "+str(accBar[j])+"\%")
        app_no='a_'+str(g_id[j])
        accuracy=str(accBar[j])+'%'
        accfinal.append((app_no, accuracy))
        
elif (adv=='OW'): #OpenWorld Settings where training and testing data are collected from different settings of same app
    data=Final_feature(SG,OW,data_process)#call original input data for training
    OW=True #OW=True state that open-world (OW) data calling is true, so get input data from OW settings
    dataOW=Final_feature(SG,OW,data_process) #collect open-world data

#loop over n number of apps
    for j in range(len(g_id)):
        d=data[opt_rp] #ratio of FBA=1
        d1=dataOW[opt_rp]
        #preprocess data
        d_h=data_preProcess(d,g_id[j],target)
        d_h1=data_preProcess(d1,g_id[j],target)
        
        #Train-Test division for both dataset
        X_train,y_train,X_test1,y_test1,X_val,y_val=train_test(d_h,M,rt,target,sk) #use this training data
        X_train1,y_train1,X_test,y_test,X_val1,y_val1=train_test(d_h1,M,rt,target,sk) #use Open world data for evaluation
        sd=index_dev(y_test) #find the 'sub session' division point
        
        #Select and train model
        if Model=='RF':
            print('Chosen model is Random Forest')
        elif Model=='XGB':
            print('Chosen model is Xboost')
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test=le.fit_transform(y_test)
        
        model=final_model(Model,SG1,cross_val, X_train,y_train)
        models.append(model) #save each model for each app
        y_pred = model.predict(np.array(X_test)) #prediction for each block
        new_pred=np.array(divide_pred(y_pred,sd),dtype=object) #divided prediction
        true_preds=np.array(divide_pred(y_test,sd),dtype=object)
        true_labels=final_label(true_preds,len(sd))
        #true_labels=final_label(np.array(divide_pred(y_test,sd),dtype=object).astype(int),len(sd))
        final=final_label(new_pred,len(sd)) #calculate final label based on 'sub session'
        accBar[j]=accuracy_score(true_labels, final)*100 #accuracy per app per ratio
        
        #collect top features
        feature=Top_Features(model,g_id[j],f_n,X_train)
        final_features.append(feature)
        #identification accuracy
        print("final identification accuracy for app a_"+str(j+1)+" is "+str(accBar[j])+"\%")
        app_no='a_'+str(g_id[j])
        accfinal.append((app_no,acc_emotion[j,:]))

#Adversarial settings=Zero day settings
elif (adv=='Zero-Day'):
    #input data
    data=Final_feature(SG,OW,data_process)#call original input data for training
    d=data[opt_rp] #data with optimized ratio
    
    for j in range(len(gname)):
        #initialize zero day models
         models_zd=[]
         
         # call jth app group
         g=app_grouping(gname[j])
         
         #train and evaluate adversaries
         # execute if there is only one application in the group
         if(len(g))==1:
            d_h=data_preProcess(d,g[0],target)
            print("check unique users id:",np.unique(d_h['user_id'].values))
            print("load train/test dataset")
            X_train,y_train,_,_,X_val,y_val=train_test(d_h,M,rt,target,sk)
            print('size of training data',X_train.shape)
             #select Model and change data according to model type
            if Model=='RF':
                print('Chosen model is Random Forest')
            elif Model=='XGB':
                print('Chosen model is Xboost')
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test=le.fit_transform(y_test)
             #call final model
            model= final_model(Model,SG,cross_val, X_train,y_train)
            models_zd.append(model)
        
        ## execute if there is more than one application in the group
         else:
         # Loop through the apps in each app group
            for k in range(len(g)):
                # List all apps in the training data, excluding one specific app
                g_k=g[:k] + g[k+1:]
                print('the app/apps that participate in training is:',g_k)
                
                #preprocess data for training
                d_h=f_data(g_k,d,target)
                print("check unique users id:",np.unique(d_h['user_id'].values))
                print("load train/test dataset")
                
                #use all sessions data as training data
                X_train1,y_train1,X_train2,y_train2,X_val,y_val=train_test(d_h,M,rt,target,sk)
                
                #merge and get final training data
                X_train = np.concatenate((X_train1, X_train2), axis=0)
                y_train = np.concatenate((y_train1, y_train2), axis=0)
                print('size of training data',X_train.shape)
                
                 #select Model and change data according to model type
                if Model=='RF':
                    print('Chosen model is Random Forest')
                elif Model=='XGB':
                    print('Chosen model is Xboost')
                    le = LabelEncoder()
                    y_train = le.fit_transform(y_train)
                    y_test=le.fit_transform(y_test)
                    
                 #call final model
                model= final_model(Model,SG,cross_val, X_train,y_train)
                models_zd.append(model)
                
         print('print length of zd models',len(models_zd))
         #collect top features
         #feature=Top_Features(model,gname[j],f_n,X_train)
         #final_features.append(feature)
         
         #evaluate all groups apps using data
         for i in range(len(gname)):
            
            # Initialize accuracy array to store evaluations of the ith model in the jth app group
            acc_i=np.zeros(len(models_zd))
            for k1 in range(len(models_zd)):
                #If there is only one app in app group or data coming from similar app group
                if(len(g))==1:
                    d_h=data_preProcess(d,g[0],target)
                
                # Check if the app group is the same as the training app group
                elif(i==j):
                    d_h=data_preProcess(d,g[k1],target)
                    
                #if there is more than one app
                else:
                    d_h=f_data(g,d,target)#evaluate for each of all app groups data on jth app group model
                _,_,X_test,y_test,_,_=train_test(d_h,M,rt,target,sk)
                print('size of test data',X_test.shape)
                sd=index_dev(y_test)
                
                # Calculate predictions for each model across all apps in every app group
                y_pred = models_zd[k1].predict(np.array(X_test)) #prediction for each block
                new_pred=np.array(divide_pred(y_pred,sd),dtype=object) #divided prediction
                true_preds=np.array(divide_pred(y_test,sd),dtype=object)
                true_labels=final_label(true_preds,len(sd))

                #final labe (per user)
                final=final_label(new_pred,len(sd)) #calculate final label based on 'sub session'
                acc_i[k1]=accuracy_score(true_labels, final)*100
            print("accuracy for each app:", acc_i)
            
            #accuracy per app group for each of all app group
            accGroup[j,i]=np.average(acc_i)
            
         print("final identification accuracy for "+gname[j]+" app group is "+str(accGroup[j,:])+"\%")
         accfinal.append((gname[j], accGroup[j,:]))
            
         
# Save to file in the output directory
#directory for identification accuracy
output_file = os.path.join(output_dir, 'output_'+Model+'_'+adv+'_adversary'+'.txt')

#directory for top features
feature_file = os.path.join(output_dir, 'feature_'+Model+'_'+adv+'_adversary'+'.txt')

#check directory path
print("check directory path",output_file)
print("check directory path",feature_file)

#np.savetxt(output_file, accfinal, header='Attack Accuracy for each app', comments='')
#save identification accuracy 
with open(output_file, 'w') as file:
             file.write(str(accfinal))

#save feature importance
with open(feature_file, 'w') as file:
             file.write(str(final_features))
             

  
  

