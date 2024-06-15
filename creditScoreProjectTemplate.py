#Scikit Template
'''created by Casey Bennett 2018, www.CaseyBennett.com
   Copyright 2018

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License (Lesser GPL) as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
'''

# cd Desktop\"Class Files"\"DSC350 Class Files"\"5272024 CreditScore"\"2. Prepared Data"
# run twice to start the first time

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import regex
import os
import csv
import math
import time
import numpy as np
from operator import itemgetter
import time
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.feature_selection import RFE, VarianceThreshold, SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif, chi2
from sklearn import metrics
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import KBinsDiscretizer, scale
from sklearn.metrics import get_scorer_names, get_scorer
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import shap
import xgboost


#Handle annoying warnings
import warnings, sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


#############################################################################
#
# Global parameters
#
#####################

target_idx=0                                        #Index of Target variable
cross_val=1                                         #Control Switch for CV
norm_target=0                                       #Normalize target switch
norm_features=0                                     #Normalize target switch
binning=0                                           #Control Switch for Bin Target
bin_cnt=2                                           #If bin target, this sets number of classes
feat_select=1                                       #Control Switch for Feature Selection
fs_type=1                                          #Feature Selection type (1=Stepwise Backwards Removal, 2=MI Scores)
lv_filter=0                                         #Control switch for low variance filter on features
feat_start=1                                        #Start column of features
k_cnt=5                                             #Number of 'Top k' best ranked features to select, only applies for fs_types 1 and 3
#Set global model parameters
rand_st=0                                           #Set Random State variable for randomizing splits on runs


#############################################################################
#
# Load Data
#
#####################
cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))
file1 = pd.read_csv('train.csv', delimiter=',', quotechar='"', dtype={'Column26': float})




#############################################################################
#
# Preprocess data
#
##########################################

# Cleaning Data & Processing Data
##Dropping Columns
columns_to_drop = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN']
file1.drop(columns=columns_to_drop, inplace=True)
###function for later use
def parse_years_and_months(age):
    if isinstance(age, str):
        age_parts = age.split(' Years and ')
        years = int(age_parts[0]) if 'Years' in age else 0 
        months_str = age_parts[1].split(' Months')[0] if 'Months' in age_parts[1] else '0' 
        months = int(months_str)
        total_months = years * 12 + months
        return total_months
    else:
        return 0  

##Cleaning Columns
file1['Age'] = file1['Age'].fillna('0').str.extract('(\\d+)').astype(float).astype(int)
file1['Num_of_Loan'] = file1['Num_of_Loan'].fillna('0').str.extract('(\\d+)').astype(float).astype(int)
file1['Num_of_Delayed_Payment'] = file1['Num_of_Delayed_Payment'].fillna('0').str.extract('(\\d+)').astype(float).astype(int)
file1['Annual_Income'] = file1['Annual_Income'].str.replace(r'[^0-9.]', '', regex=True)
file1['Annual_Income'] = file1['Annual_Income'].astype(float)
file1['Changed_Credit_Limit'] = file1['Changed_Credit_Limit'].replace('_', np.nan)
file1['Changed_Credit_Limit'] = pd.to_numeric(file1['Changed_Credit_Limit'], errors='coerce')
file1['Changed_Credit_Limit'] = file1['Changed_Credit_Limit'].fillna(0)
file1['Outstanding_Debt'] = file1['Outstanding_Debt'].astype(str)
file1['Outstanding_Debt'] = pd.to_numeric(file1['Outstanding_Debt'], errors='coerce')
file1['Outstanding_Debt'] = file1['Outstanding_Debt'].fillna(0)
file1['Amount_invested_monthly'] = file1['Amount_invested_monthly'].astype(str)
file1['Amount_invested_monthly'] = file1['Amount_invested_monthly'].replace('', '0')
file1['Amount_invested_monthly'] = file1['Amount_invested_monthly'].str.replace(r'[^0-9.]', '')
file1['Amount_invested_monthly'] = pd.to_numeric(file1['Amount_invested_monthly'], errors='coerce')
file1['Monthly_Balance'] = file1['Monthly_Balance'].astype(str)
file1['Monthly_Balance'] = file1['Monthly_Balance'].str.replace(r'[^0-9.-]+', '')
file1['Monthly_Balance'] = pd.to_numeric(file1['Monthly_Balance'], errors='coerce')
file1['Monthly_Balance'] = file1['Monthly_Balance'].fillna(0)
file1['Amount_invested_monthly'] = file1['Amount_invested_monthly'].fillna(0)
file1['Credit_History_Age_Months'] = file1['Credit_History_Age'].apply(parse_years_and_months)

##outlier handling
selected_columns_file1 = file1[['Num_Bank_Accounts', 'Interest_Rate', 'Annual_Income', 'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Total_EMI_per_month', 'Num_of_Loan', 'Num_Credit_Card']]
percentile_threshold = 0.98
percentiles = selected_columns_file1.quantile(percentile_threshold)
for column in selected_columns_file1.columns:
    file1 = file1[file1[column] <= percentiles[column]]

##filtering garbage out, blank spaces and junk
file1 = file1[file1['Payment_Behaviour'] != '!@9#%8']
file1 = file1[file1['Occupation'] != '_______']
file1 = file1[file1['Credit_Mix'] != '_']
#removing negative values
selected_columns = ['Delay_from_due_date', 'Changed_Credit_Limit', 'Num_Bank_Accounts']
for column in selected_columns:
    file1 = file1[file1[column] >= 0]

##dropping a few more columns not useful to analysis 
moreColumnsToDrop = ['Credit_History_Age','Monthly_Inhand_Salary', 'Type_of_Loan']
file1.drop(columns=moreColumnsToDrop, inplace=True)


# Decreasing Ranges in Standard Deviations
##Involves deleting "BS" entries, such as age in the thousands. Despite the inital wipe, lots of outliers are beyond standard deviation.
file1 = file1[file1['Age'] < 60]
file1 = file1[file1['Num_Credit_Card'] <= 10]
file1 = file1[file1['Interest_Rate'] <= 50]
file1 = file1[file1['Num_of_Loan'] <= 12]
file1 = file1[file1['Num_Bank_Accounts'] <= 10]
file1 = file1[file1['Delay_from_due_date'] <= 60]
file1 = file1[file1['Changed_Credit_Limit'] <= 30]
file1 = file1[file1['Num_Credit_Inquiries'] <= 12]
file1 = file1[file1['Total_EMI_per_month'] <= 200]
file1 = file1[file1['Outstanding_Debt'] <= 1500]

##encoding
categories = ['Poor', 'Standard', 'Good']
encoder = OrdinalEncoder(categories=[categories])
file1['Credit_Score_Encoded'] = encoder.fit_transform(file1[['Credit_Score']])
label_encoder = LabelEncoder()
file1['Occupation_Encoded'] = label_encoder.fit_transform(file1['Occupation'])
categories = ['Bad', 'Standard', 'Good']
encoder = OrdinalEncoder(categories=[categories])
file1['Credit_Mix_Encoded'] = encoder.fit_transform(file1[['Credit_Mix']])
categories_payment_behaviour = [
    'Low_spent_Small_value_payments', 
    'Low_spent_Medium_value_payments', 
    'Low_spent_Large_value_payments', 
    'High_spent_Small_value_payments', 
    'High_spent_Medium_value_payments', 
    'High_spent_Large_value_payments'
]
encoder_payment_behaviour = OrdinalEncoder(categories=[categories_payment_behaviour])
file1['Payment_Behaviour_Encoded'] = encoder_payment_behaviour.fit_transform(file1[['Payment_Behaviour']])
columns_to_drop = [ 'Payment_Behaviour', 'Credit_Mix', 'Occupation','Credit_Score']
file1.drop(columns=columns_to_drop, inplace=True)

#missing value check
total_missing_values = file1.isnull().sum().sum()

if total_missing_values == 0:
    print("There are no missing values!")
else:
    print("Total missing values:", total_missing_values)

#############################################################################
#
# Feature Selection
#
##########################################

# Feature Engineering
##Calculate the total number of accounts (Bank Accounts + Credit Cards)
file1['Total_Num_Accounts'] = file1['Num_Bank_Accounts'] + file1['Num_Credit_Card']
##Calculate the total outstanding debt per account
file1['Debt_Per_Account'] = file1['Outstanding_Debt'] / file1['Total_Num_Accounts']
##Calculate the ratio of outstanding debt to annual income
file1['Debt_to_Income_Ratio'] = file1['Outstanding_Debt'] / file1['Annual_Income']
##Calculate the total number of delayed payments per account
file1['Delayed_Payments_Per_Account'] = file1['Num_of_Delayed_Payment'] / file1['Total_Num_Accounts']
##Calculate the total monthly expenses (EMI + Monthly Investments) 
file1['Total_Monthly_Expenses'] = file1['Total_EMI_per_month'] + file1['Amount_invested_monthly']
##remove payment from min amount to see if it fixes string issues
columns_to_drop = ['Payment_of_Min_Amount']
file1.drop(columns=columns_to_drop, inplace=True)

# Split into data and target sets
cols = [16]
target = file1[file1.columns[cols]]
cols = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24]
df = file1[file1.columns[cols]]
print(df.columns.tolist())


# Feature Selection
if feat_select==1:
    '''Three steps:
       1) Run Feature Selection
       2) Get lists of selected and non-selected features
       3) Filter columns from original dataset
       '''
    
    print('--FEATURE SELECTION ON--', '\n')

    if fs_type==1:
        #backwards feature selection
        warnings.filterwarnings('ignore')
        backward_feature_selection = SequentialFeatureSelector(RandomForestClassifier(n_estimators=20, min_samples_split=3, n_jobs=-1,
        random_state=rand_st), k_features=5, forward=False, floating=False, verbose=5, scoring="accuracy", cv=2).fit(df,target)
        print("\nSelected Features:")
        selectedFeatures = backward_feature_selection.k_feature_names_
        print(selectedFeatures)
        df = df[[i for i in selectedFeatures if i in df.columns]] #extract only colNames features, aka features from feature selection
        print('Feature Selection has been applied to df object!')
        
        #set df to be only the colNames
        
    if fs_type==2:
        warnings.filterwarnings('ignore')
        #MI Scores
        ##Encoding MI Scores
        categorical_columns = df.select_dtypes(include=['object']).columns
        data_encoded = file1.copy()
        encoder = OrdinalEncoder()
        data_encoded[categorical_columns] = encoder.fit_transform(data_encoded[categorical_columns])
        y = data_encoded['Credit_Score_Encoded']
        X = data_encoded.drop(columns=['Credit_Score_Encoded'])

        ##MI Scores
        mi_scores = mutual_info_classif(X, y)
        sorted_mi_scores = sorted(zip(X.columns, mi_scores), key=lambda x: x[1], reverse=True)
        sorted_columns = [x[0] for x in sorted_mi_scores]
        sorted_scores = [x[1] for x in sorted_mi_scores]
       

        ##Seaborn Display
        sns.set_style('darkgrid')
        miPlot = sns.barplot(x=sorted_columns, y=sorted_scores, hue = sorted_columns, palette = 'cool', legend=False).set_title("Features' Mutual Information Score")
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('MI Score')
        plt.show()
        
        #Lists of selected and nonselected features
        selectedFeatures = []
        unselectedFeatures = []
        for feature, score in sorted_mi_scores:
            if(score > 0.2):
                selectedFeatures.append(feature)
            else:
                unselectedFeatures.append(feature)
        print('Selected:', selectedFeatures)
        print('Features (total/selected):', (len(selectedFeatures) + len(unselectedFeatures)), len(selectedFeatures))
        print('\n')

        #Filter Columns from original dataset
        df.drop(columns=unselectedFeatures, inplace=True)
        print('Feature Selection has been applied to df object!')
        
#############################################################################
#
# Train SciKit Models
#
##########################################

#train test split
df_train, df_test, target_train, target_test = train_test_split(df, target, test_size=0.3, random_state=rand_st) #train test split
#Setup Crossval regression scorers

#Random Forest Regressor Model
start_ts=time.time()
model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=3, criterion='friedman_mse', bootstrap=True, random_state=rand_st)
##scores 
if cross_val == 0:
    ##baseline model comparison
    print("Running Basic Random Forest Model Comparison")
    model.fit(df_train, target_train)
    y_pred = model.predict(df_test)
    mse = mean_squared_error(target_test, y_pred)
    baseline_prediction = target_train.mean()  
    baseline_mse = mean_squared_error(target_test, [baseline_prediction] * len(target_test))
    print("Baseline MSE:", baseline_mse)
    print("MSE RFRegressor", mse)
    print("Comparison Runtime:", time.time()-start_ts)
if cross_val == 1:
    print("Running Random Forest Cross Validation!")
    scorers = {'Neg_MSE': 'neg_mean_squared_error', 'expl_var': 'explained_variance'}
    scores = cross_validate(model, X = df_train, y=target_train, scoring=scorers, cv=5)
    scores_RMSE = np.asarray([math.sqrt(-x) for x in scores['test_Neg_MSE']])                                       #Turns negative MSE scores into RMSE
    scores_Expl_Var = scores['test_expl_var']
    print("CV Random Forest RMSE:: %0.2f (+/- %0.2f)" % ((scores_RMSE.mean()), (scores_RMSE.std() * 2)))
    print("CV Random Forest Expl Var: %0.2f (+/- %0.2f)" % ((scores_Expl_Var.mean()), (scores_Expl_Var.std() * 2))) 
    print("CV Runtime:", time.time()-start_ts)
    #needs updating


#xgboost/gradient boost
if cross_val == 0: 
    start_ts=time.time()
    xgb_model = XGBClassifier(n_estimators=3500, learning_rate=0.05, random_state=77)
    xgb_model.fit(df_train, target_train)
    y_pred_xgb = xgb_model.predict(df_test)
    mse = mean_squared_error(target_test, y_pred_xgb)
    print("Mean Squared Error (MSE):", mse)                           
    print("XGBoost Runtime:", time.time()-start_ts)
#gradient boost
if cross_val == 1:
    start_ts=time.time()
    clf= GradientBoostingClassifier(n_estimators=100, loss='log_loss', learning_rate=0.1, max_depth=3, min_samples_split=3, random_state=rand_st)
    scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'} 
    scores= cross_validate(estimator=clf, X= df_train, y= target_train, scoring=scorers, cv=5)  
    scores_Acc = scores['test_Accuracy']                                                                                                                                    
    print("Gradient Boost Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
    scores_AUC= scores['test_roc_auc']                                                                     #Only works with binary classes, not multiclass                  
    print("Gradient Boost AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
    print("CV Runtime:", time.time()-start_ts)


#neural network
start_ts=time.time()
scorers = {'Accuracy': 'accuracy', 'roc_auc': 'roc_auc'} 
nnClf= MLPClassifier(activation='logistic', solver='lbfgs', alpha=0.0001, max_iter=1000, hidden_layer_sizes=(10,), random_state=rand_st)
scores= cross_validate(estimator=nnClf, X= df_train, y= target_train, scoring=scorers, cv=5)  

scores_Acc = scores['test_Accuracy']                                                                                                                                    
print("Neural Network Acc: %0.2f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))                                                                                                    
scores_AUC= scores['test_roc_auc']                                                                     #Only works with binary classes, not multiclass                  
print("Neural Network AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))                           
print("CV Runtime:", time.time()-start_ts)

#############################################################################
#
# Model Validation & Synthetic Sampling
#
##########################################

#SMOTE Sampling
smote_params = {
    'sampling_strategy': 'auto',  
    'random_state': rand_st,           
    'k_neighbors': 5,             
    'n_jobs': -1                  
}
#runtime
start_ts=time.time()
#smote train test split
smote = SMOTE(**smote_params)
X_train = df
y_train = target
X_smote, y_smote = smote.fit_resample(X_train, y_train)
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_smote, y_smote, test_size=0.15, random_state=77)
#rf
rf_classifier = RandomForestClassifier(n_estimators=500, bootstrap=True)
rf_classifier.fit(X_train_smote, y_train_smote)
print("Accuracy on training set:", rf_classifier.score(X_train_smote, y_train_smote))
y_pred_smote = rf_classifier.predict(X_test_smote)
accuracy_smote = accuracy_score(y_test_smote, y_pred_smote)

print("Accuracy on SMOTE test set:", accuracy_smote)
print("SMOTE Runtime:", time.time()-start_ts)

#rf accuracy
start_ts=time.time()
y_pred = rf_classifier.predict(df_train)
accuracy = accuracy_score(target_train, y_pred)
print("Accuracy on original test set:", accuracy)
##Confusion Matrix
matrix = confusion_matrix(target_train, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(matrix, annot=True, cbar=False, cmap='Pastel2', linewidth=0.5, fmt="d")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix for RandomForestClassifier on original test set')
#SMOTE Test Set
matrix = confusion_matrix(y_test_smote, y_pred_smote)
plt.figure(figsize=(6, 6))
sns.heatmap(matrix, annot=True, cbar=False, cmap='Pastel2', linewidth=0.5, fmt="d")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix for RandomForestClassifier on SMOTE Test Set')
##Output
print('\nClassification report for SMOTE test set:\n', classification_report(y_test_smote, y_pred_smote))
print('\nClassification report for original test set:\n', classification_report(target_train, y_pred))
print("Classification Matrix Runtime:", time.time()-start_ts)
plt.show()

#SHAP Interpretation
##XGBoost Shap
start_ts = time.time()
xgClassifier = XGBClassifier(objective='multi:softmax')
xgClassifier.fit(df_train, target_train)
print('First entry predicted probabilities',xgClassifier.predict_proba(df)[0])
explainer = shap.Explainer(xgClassifier)
shap_values = explainer(df_train)
print(np.shape(shap_values.values)) #Amount of SHAP Value Categories
print("XG SHAP Runtime:", time.time()-start_ts)
#waterfall plot for first observation
#THESE CAN RANDOMLY FAIL TO RUN!
shap.plots.waterfall(shap_values[0, :, 0])
shap.plots.waterfall(shap_values[0, :, 1])
shap.plots.waterfall(shap_values[0, :, 2])

#shap value prediction loop
xgPred = xgClassifier.predict(df_train)
new_shap_values = []

for i, pred in enumerate(xgPred):
    #get shap values for predicted class
    new_shap_values.append(shap_values.values[i][:, pred])
#replace shap values
shap_values.values = np.array(new_shap_values)
print(shap_values.shape)

#SHAP Plots
shap.plots.beeswarm(shap_values)
shap.plots.bar(shap_values)