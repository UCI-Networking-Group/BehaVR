#model
#@ijarin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
# Create a series containing feature importances from the model and feature names from the training data
from textwrap import wrap
from matplotlib.lines import Line2D


#Choose best RF model
def RF_tuning(X_train,y_train,cv):
    param_dist = {'n_estimators': randint(50,200),
              'max_depth': randint(1,20)}

# Create a random forest classifier
    rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(rf,
                                 param_distributions = param_dist,
                                 n_iter=10,
                                 cv=cv)

# Fit the random search object to the data
    rand_search.fit(X_train, y_train)
    # Create a variable for the best model
    best_rf = rand_search.best_estimator_

    # Print the best hyperparameters
    print('Best hyperparameters:',  rand_search.best_params_)
    return best_rf

#Choose XGB Model
def XB_tuning(X_train,y_train,cv):
  # Define a grid of hyperparameters to search
  param = {
    'n_estimators': [50,100,150,200],  # Number of trees
    'max_depth': [3,5, 10,15,20] ,      # Maximum depth of trees
    'learning_rate': [0.1, 0.01],  # Learning rate
    #'min_child_weight': [1, 2, 3],  # Minimum sum of instance weight needed in a child
    }
  xgb_model = xgb.XGBClassifier()
  # Create the GridSearchCV object
  grid_search = GridSearchCV(estimator=xgb_model, param_grid=param, scoring='accuracy', cv=cv)

  # Fit the grid search to the data
  grid_search.fit(X_train, y_train)

  # Get the best hyperparameters
  best_params = grid_search.best_params_
  print("Best Hyperparameters:", best_params)

  # Train the XGBoost model with the best hyperparameters
  best_xgb_model = xgb.XGBClassifier(**best_params)
  return best_xgb_model

def final_model(Model,SG, cross_val, X_train,y_train):
    if Model=='RF':
            print('Chosen model is Random Forest')
            if SG=='HJ':
                print('handling Hand-data')
                model_RF = RandomForestClassifier()
                model_RF.fit(np.array(X_train),y_train)
                feature_importances = model_RF.feature_importances_
                top_400_indices = feature_importances.argsort()[-400:][::-1] #Choose top 400 features
                X_train = X_train.iloc[:, top_400_indices]
                X_test=X_test.iloc[:, top_400_indices]
            #model= RandomForestClassifier()
            model=RF_tuning(X_train,y_train,cross_val)
            model.fit(np.array(X_train),y_train)
    elif Model=='XGB':
            if dtype=='hand':
                print('handling Hand-data')
                model_XB = xgb.XGBClassifier()
                model_XB.fit(np.array(X_train),y_train)
                feature_importances = model_XB.feature_importances_
                top_400_indices = feature_importances.argsort()[-400:][::-1]
                X_train = X_train.iloc[:, top_400_indices]
                X_test=X_test.iloc[:, top_400_indices]
            #tune and find best model
            model=XB_tuning(X_train,y_train,cross_val)
            model.fit(np.array(X_train),y_train)
    return model
    
#Get Top features
def Top_Features(final_model,app_id,f_n,X_train_app):
        # Define the list to store feature importance data
        feature_importance_final = []
        
        #calculate each features importance score for model
        feature_importances = pd.Series(final_model.feature_importances_, index=X_train_app.columns).sort_values(ascending=False)
        
        #store top f_n features
        feature_top = feature_importances.head(f_n)
        f2=feature_top.index
        f2=f2.str.replace('Pos0', 'Position.x')
        f2=f2.str.replace('Pos1', 'Position.y')
        f2=f2.str.replace('Pos2', 'Position.z')

        # Append group_id and app_id to feature_importance_final
        feature_importance_final.append("App ID: " + "a_"+str(app_id) + " ")
        
        # Append feature names and their importance values to feature_importance_final
        for feature, importance in feature_top.items():
            #feature_importance_final.append(f"{feature}: {importance}\n") #if we want both top features and their values
            feature_importance_final.append(f"{feature}") #Only store feature names

        return feature_importance_final



    
