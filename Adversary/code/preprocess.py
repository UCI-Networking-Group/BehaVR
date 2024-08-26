#@ijarin
#necessary function to preprocess input data, devide train/test, app grouping etc
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

#Function for Preprocessing Body Motion Data
def change_nameH(datar):
  datar.columns=datar.columns.str.replace('_1', ' Headset')
  datar.columns=datar.columns.str.replace('_2', ' Left Controller')
  datar.columns=datar.columns.str.replace('_3', ' Right Controller')
  datar.columns=datar.columns.str.replace('Pos0', 'Position.x')
  datar.columns=datar.columns.str.replace('Pos1', 'Position.y')
  datar.columns=datar.columns.str.replace('Pos2', 'Position.z')
  datar.columns=datar.columns.str.replace('_max', ' Max')
  datar.columns=datar.columns.str.replace('_min', ' Min')
  datar.columns=datar.columns.str.replace('_mean', ' Mean')
  datar.columns=datar.columns.str.replace('_median', ' Median')
  datar.columns=datar.columns.str.replace('_std', ' Std')
  return datar
  
#Function for Processing Eye Gaze Data
def change_nameE(datar):
  datar.columns=datar.columns.str.replace('_x', '_L')
  datar.columns=datar.columns.str.replace('_y', '_R')
  return datar

#Change name of final eyedata (after augmentation)
def change_nameEye(datar):
  datar.columns=datar.columns.str.replace('_LR', ' Left Right')
  datar.columns=datar.columns.str.replace('_L', ' Left')
  datar.columns=datar.columns.str.replace('_R', ' Right')
  datar.columns=datar.columns.str.replace('_max', ' Max')
  datar.columns=datar.columns.str.replace('_min', ' Min')
  datar.columns=datar.columns.str.replace('_mean', ' Mean')
  datar.columns=datar.columns.str.replace('_median', ' Median')
  datar.columns=datar.columns.str.replace('_std', ' Std')
  return datar
#Feature Augmentation of Eye data
def RLeft_Right(d, b=1):
    metrics = [
        'Quaty_mean', 'Quatx_mean', 'Quatw_mean',
        'Quaty_min', 'Quatx_min', 'Quatw_min',
        'Quaty_max', 'Quatx_max', 'Quatw_max',
        'Quaty_std', 'Quatx_std', 'Quatw_std',
        'Quaty_median', 'Quatx_median', 'Quatw_median'
    ]
    
    for metric in metrics:
        d[f'{metric}_LR'] = abs(d[f'{metric}_L']*b - d[f'{metric}_R']*b)
    
    return d
  
#Function for Preprocessing Facial Data
def change_nameFace(datar):
  for i in range(64,0,-1):
      s2='Element['+str(i)+']_'
      s1='w'+str(i-1)+'_'
      #print(s1,s2)
      datar.columns=datar.columns.str.replace(s1,s2)

  datar.columns=datar.columns.str.replace('_max', ' Max')
  datar.columns=datar.columns.str.replace('_min', ' Min')
  datar.columns=datar.columns.str.replace('_mean', ' Mean')
  datar.columns=datar.columns.str.replace('_med', ' Median')
  datar.columns=datar.columns.str.replace('_std', ' Std')
  return datar

def change_nameF(datar):
  for i in range(64,0,-1):
      s2='w'+str(i)+'_'
      s1='w'+str(i-1)+'_'
      #print(s1,s2)
      datar.columns=datar.columns.str.replace(s1,s2)
  return datar
  
  
#Process Hand Data/Feature Engineering
def change_nameHand(datar):
  datar.columns=datar.columns.str.replace('_x', ' Left')
  datar.columns=datar.columns.str.replace('_y', ' Right')
  datar.columns=datar.columns.str.replace('BoneRotation', 'Rotation')
  datar.columns=datar.columns.str.replace('BonePosition', 'Position')
  for i in range(26,0,-1):
      s1='x'+'_'+str(i-1)
      s2='y'+'_'+str(i-1)
      s3='z'+'_'+str(i-1)
      s4='w'+'_'+str(i-1)

      #print(s1,s2)
      s11='x'+'['+str(i)+']'
      s22='y'+'['+str(i)+']'
      s33='z'+'['+str(i)+']'
      s44='w'+'['+str(i)+']'
      datar.columns=datar.columns.str.replace(s1,s11)
      datar.columns=datar.columns.str.replace(s2,s22)
      datar.columns=datar.columns.str.replace(s3,s33)
      datar.columns=datar.columns.str.replace(s4,s44)

  datar.columns=datar.columns.str.replace('_max', ' Max ')
  datar.columns=datar.columns.str.replace('_min', ' Min ')
  datar.columns=datar.columns.str.replace('_med', ' Med ')
  datar.columns=datar.columns.str.replace('_std', ' Std ')
  datar.columns=datar.columns.str.replace('_mean', ' Mean ')

  datar=datar.sort_values(by='user_id')
  datar = datar.reset_index(drop=True)
  return datar

#eliminate specific types of features (here Headset Features)
def change_eliminate_Head(datar):
  datar.columns=datar.columns.str.replace('_1', '_H')
  datar.columns=datar.columns.str.replace('_2', '_CL')
  datar.columns=datar.columns.str.replace('_3', '_CR')
  datar.columns=datar.columns.str.replace('0', 'x')
  datar.columns=datar.columns.str.replace('1', 'y')
  datar.columns=datar.columns.str.replace('2', 'z')
  filtered_columns = datar.filter(like='_H', axis=1).columns
  datar = datar.drop(columns=filtered_columns)
  return datar

#eliminate specific types of features (here Right Hand Features)
def change_eliminate_RightHand(datar):
  filtered_columns = datar.filter(like='Right', axis=1).columns
  datar = datar.drop(columns=filtered_columns)
  return datar
  
#eliminate specific types of features (here Emotion Facial Expression Features)
def change_nameFacialEexpression(datar,f):
  for i in range(64,0,-1):
      s2='w'+str(i)+'_'
      s1='w'+str(i-1)+'_'
      #print(s1,s2)
      datar.columns=datar.columns.str.replace(s1,s2)
  datar=datar.drop(f,axis=1)
  return datar

#data preprocess for app adversary
def data_preProcess(d_h,g_id,target):
  #identify the columns that contains all zero values
   d_h=d_h.fillna(0)
   drop_list=[]

   for col in d_h.columns:
       if (d_h[col] == 0).all():
         drop_list.append(col)
       if (d_h[col] == 1).all():
          drop_list.append(col)

   d_h=d_h.drop(['block_id'],axis=1)
   #value_counts = Counter(d_h['user_id'])
   #print(value_counts)
   d_h=d_h.sort_values(by='user_id')
   d_h = d_h.reset_index(drop=True)
   if (target=='user_id'):         #select data according to game id
      d_h=d_h[d_h['game_id']== g_id]
      d_h=d_h.drop(['game_id'],axis=1)
   else:
      d_h=d_h.drop('user_id',axis=1)
   return d_h
   
   
#divide the dataset into training and testing based on their round. For round-1: training data, round-2: test data

#Get d_train and d_test from a function
def dT(d_h,rt):
  if (rt=='t'):
    #d_h = d_h[d_h.user_id= P]
    d_h = d_h[(d_h['user_id'].isin(P))]
  d_train = d_h[d_h['round_id']== 1]  #round_1=train-data
  d_test = d_h[d_h['round_id']== 2]
  d_train=d_train.drop(['round_id'],axis=1)
  d_test=d_test.drop(['round_id'],axis=1)
    #round_2=test data
  return d_train, d_test

#stradified train test split/considering subsession
def stratified_train_test_split(X, Y, M):
    X=np.array(X)
    splits = {}
    for i in range(M):
        K = M - i
        if K >= 2:
            skf = StratifiedKFold(n_splits=K, shuffle=False)
            split_indices = []
            for train_index, test_index in skf.split(X, Y):
                split_indices.append((train_index, test_index))
            splits[i] = split_indices
            X, X_test = X[split_indices[0][1]], X[split_indices[0][0]]
            Y, Y_test = Y[split_indices[0][1]], Y[split_indices[0][0]]
        elif K == 1:
            splits[i] = [(range(len(X)), [])]
            X_test, Y_test = X, Y
        splits[i][0] = (range(len(X)), [])
        X_train, Y_train = X, Y
    return X_train,Y_train

#train_test_split
def train_test(d_h,M,rt,target):
  d_train,d_test=dT(d_h,rt)
  y_train = np.array(d_train[target])
  #print(np.unique(y_train))
  X_train= d_train.drop(target, axis = 1)
  y_test = np.array(d_test[target])
  #print(y_test)
  X_test= d_test.drop(target, axis = 1)
  #print(X_test.shape)
  #X_test,x,y_test,y=train_test_split(X_test, y_test, test_size=1, random_state=0,shuffle=False)
  #X_test,y_test=split(X_test,y_test,20)
  X_test,y_test = stratified_train_test_split(X_test,y_test,M)
  X_train,X_val,y_train,y_val=train_test_split(X_train, y_train, test_size=0.0000001, random_state=0)
  #print(X_test.shape)
  #print("shape of train data", X_train.shape,y_train.shape)
  #print("shape of test data", X_test.shape,y_test.shape)
  return X_train, y_train, X_test, y_test,X_val, y_val


#functions for Producing final labels
def index_dev(arr):
  arr=np.array(arr)
  ind = np.where(arr[:-1] != arr[1:])[0] + 1
  return ind

def divide_pred(arr,points):
  #points=mdfs(points)
  new_arr = []
  start = 0
  for point in points:
      new_arr.append((arr[start:point]))
      start = point
      #print(new_arr)
  new_arr.append(arr[start:])
  return new_arr
  
'''
def divide_pred(arr,points):
  new_arr = []
  start = 0
  for point in points:
      new_arr.append((arr[start:point]))
      start = point
  new_arr.append(arr[start:])
  return new_arr

def final_label(new_arr, n):
    label = np.zeros(n+1, dtype=int)  # Ensure label array is of int data type
    for i in range(n+1):
        counts = np.bincount(new_arr[i].astype(int))  # Convert to int dtype before bincount
        label[i] = np.argmax(counts)
    return label
'''
def final_label(new_arr, n):
    label = np.zeros(n+1, dtype=int)  # Ensure label array is of int data type
    for i in range(n+1):
        counts = np.bincount(new_arr[i].astype(int))  # Convert to int dtype before bincount
        label[i] = np.argmax(counts)
    return label
        
def final_Block(new_arr,n):
  #n=number of index
  label=np.zeros(n+1)
  for i in range(n+1):
    counts = np.bincount(new_arr[i])
    label[i] = np.argmax(counts)
    #print(counts,label[i])
  return label

def concatenate_arrays(new_preds, new_pred1):
    result = []
    for a1, a2 in zip(new_preds, new_pred1):
        if isinstance(a1, np.ndarray) and isinstance(a2, np.ndarray):
            new_array = np.concatenate((a1, a2), axis=None)
            result.append(new_array)
        elif isinstance(a1, np.ndarray):
            result.append(a1.astype(np.object))  # Convert to object dtype
        elif isinstance(a2, np.ndarray):
            result.append(a2.astype(np.object))  # Convert to object dtype
    return result

def Emotion_units():
        #Emotion Units
    #smile
    f_smile=['user_id', 'game_id', 'round_id', 'device_id', 'block_id','w33_max',
    'w33_min','w33_mean','w33_std','w33_med','w34_max',
    'w34_min','w34_mean','w34_std','w34_med','w6_max','w6_min','w6_mean','w6_std','w6_med','w5_max','w5_min','w5_mean','w5_std','w5_med']
    #surprise
    f_surprise=['user_id', 'game_id', 'round_id', 'device_id', 'block_id','w58_max','w58_min','w58_mean','w58_std','w58_med','w59_max','w59_min',
                'w59_mean','w59_std','w59_med','w60_max','w60_min','w60_mean','w60_std','w60_med','w61_max','w61_min','w61_mean','w61_std','w61_med',
    'w23_max','w23_min','w23_mean','w23_std','w23_med','w24_max','w24_min','w24_mean','w24_std','w24_med','w25_max','w25_min',
    'w25_mean','w25_std','w25_med']   #1 + 2 + 5 + 26. #(23,24)+ (58,59)+ (60,61)+ (25)
    #Anger=4 + 5 + 7 + 23  (1,2)+ (60,61)+ (29,30)+ (49,50)
    f_anger=['user_id', 'game_id', 'round_id', 'device_id', 'block_id','w1_min','w1_mean','w1_std','w1_med','w1_max',
    'w2_min','w2_mean','w2_std','w2_med','w60_max','w60_min','w60_mean','w60_std','w60_med','w61_max','w61_min','w61_mean','w61_std','w61_med',
    'w29_max','w29_min','w29_mean','w29_std','w29_med',
    'w30_max','w30_min','w30_mean','w30_std','w30_med','w49_max','w49_min','w49_mean','w49_std','w49_med','w50_max','w50_min','w50_mean','w50_std','w50_med']
    #Sadness 1 + 4 + 15; (23,24)+ (1,2)+ (31,32)
    f_sadness=['user_id', 'game_id', 'round_id', 'device_id', 'block_id','w1_max',
    'w1_min','w1_mean','w1_std','w1_med','w2_max','w2_min','w2_mean','w2_std','w2_med',
    'w23_max','w23_min','w23_mean','w23_std','w23_med','w24_max','w24_min','w24_mean','w24_std','w24_med',
    'w31_max','w31_min','w31_mean','w31_std','w31_med','w32_max','w32_min','w32_mean','w32_std','w32_med',
    ]
    f_fear=['user_id', 'game_id', 'round_id', 'device_id', 'block_id',
            'w23_max','w23_min','w23_mean','w23_std','w23_med','w24_max','w24_min','w24_mean','w24_std','w24_med',
            'w58_max','w58_min','w58_mean','w58_std','w58_med','w59_max','w59_min','w59_mean','w59_std','w59_med',
            'w1_min','w1_mean','w1_std','w1_med','w2_max','w2_min','w2_mean','w2_std','w2_med',
            'w60_max','w60_min','w60_mean','w60_std','w60_med','w61_max','w61_min','w61_mean','w61_std','w61_med',
            'w29_max','w29_min','w29_mean','w29_std','w29_med','w30_max','w30_min','w30_mean','w30_std','w30_med',
            'w43_max','w43_min','w43_mean','w43_std','w43_med','w44_max','w44_min','w44_mean','w44_std','w44_med',
            'w25_max','w25_min','w25_mean','w25_std','w25_med']

    f_disgust=['user_id', 'game_id', 'round_id', 'device_id', 'block_id','w31_max',
    'w31_min','w31_mean','w31_std','w31_med','w32_max','w32_min','w32_mean','w32_std','w32_med','w52_max','w52_min','w52_mean','w52_std','w52_med',
               'w53_max','w53_min','w53_mean','w53_std','w53_med', 'w56_max','w56_min','w56_mean','w56_std','w56_med',
               'w57_max','w57_min','w57_mean','w57_std','w57_med']
    f_contempt=['user_id', 'game_id', 'round_id', 'device_id', 'block_id','w33_max',
    'w33_min','w33_mean','w33_std','w33_med','w12_max','w12_min','w12_mean','w12_std','w12_med','w11_max','w11_min','w11_mean','w11_std','w11_med']

    f_imp=['w51_max',
    'w51_min','w51_mean','w51_std','w51_med','w28_max','w28_min','w28_mean','w28_std','w28_med','w13_max','w13_min','w13_mean','w13_std','w13_med',
           'w14_max','w14_min','w14_mean','w14_std','w14_med']


    #All
    common_words = set(f_smile).intersection(f_anger, f_sadness)
    all_words = set(f_smile + f_anger + f_sadness+f_surprise+f_fear+f_disgust)
    all_words1=set(f_smile + f_anger + f_sadness+f_surprise+f_fear+f_disgust+f_contempt+f_imp)
    words_to_keep=['user_id', 'game_id', 'round_id', 'device_id', 'block_id']
    f=list(all_words)
    f1=list(set(all_words1) - set(words_to_keep))
    emo=[f_smile,f_surprise,f_anger,f_disgust,f_fear,f_sadness, f]
    return emo, f1

#feature Engineering for face data/this will be more automated with more dataset
#take only the features based on emotion recognition/ smile=6+12
def Emotion(d,f):
  d=d[f]
  return d

#emotion process
def DataE(D,f):
  dE=[]
  for i in range(len(D)):
    d=D[i]
    dnew=Emotion(d,f)
    dE.append(dnew)
  D=dE
  return D

#app grouping
def app_groups_name():
     gname=['social','flight','Rhy','golf','IN','KW','Rhy','shoot','teleport']
     return gname

def app_grouping(app_group):
    if (app_group=='social'):
        g=[12,15,18]
    elif (app_group=='flight'): #Flight Simulation
        g=[20,19,3]
    elif (app_group=='golf'): #Golfing
        g=[6]
    elif (app_group=='IN'): #Interactive navigation
        g=[2,9,10,16,17]
    elif (app_group=='KW'): #knuckle walking
        g=[7]
    elif (app_group=='Rhy'): #Rhythm
        g=[1]
    elif (app_group=='shoot'): #Shooting and archary
        g=[5,13,14]
    elif (app_group=='teleport'): #Teleportation
        g=[4,8]
    return g

#concatenate different apps data or user can just use one apps data, in that case len(g1) should be 1
def f_data(g1,d1,target):
    M=len(g1)
    print(M)
    df_list = []
    if (M==1):
        d_h=data_preProcess(d1, g1[0],target)
    else:
        for i in range(M):
            df = data_preProcess(d1, g1[i],target)
            df = df.sample(frac=1/M)
            df_list.append(df)
        d_h = pd.concat(df_list, axis=0)
    d_h=d_h.sort_values('user_id')
    return d_h





