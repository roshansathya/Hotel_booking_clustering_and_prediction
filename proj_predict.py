#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 02:02:29 2018

@author: roshan
"""

import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('sample.csv')

len(df) - df['user_id'].nunique()

##We have 11,137 repeat customers

user_grpd = df.groupby('user_id').size()
user_grpd[user_grpd > 1].sort_values(ascending = False)

highest_user = df[df['user_id'] == 783124]

#-------------------------------------------------------------------------------------------
#Dealing with NULL values

df.drop('Unnamed: 0', axis=1,inplace=True)
df = df.dropna(how='all')
df.isnull().sum()

#Cheching for 3 business logics:
# 1. Booking date < check in date
# 2. Check in date < Check out date
# 3. Adults = 0 and child > 0   
    
df = df.drop((df[df["date_time"] > df['srch_ci']]).index,axis=0)

df = df.drop((df[df["srch_ci"] > df['srch_co']]).index,axis=0)

df[(df['srch_adults_cnt'] == 0) & (df['srch_children_cnt'] > 0)][['is_booking','srch_adults_cnt','srch_children_cnt','orig_destination_distance']]

df = df.drop((df[(df['srch_adults_cnt'] == 0) & (df['srch_children_cnt'] > 0)]).index,axis = 0)

df[df['orig_destination_distance'].isnull()]['posa_continent'].unique()
df[df['orig_destination_distance'].isnull()]['hotel_continent'].unique()

#Finding mean distance between continents to impute inplace of null values
df2 = df[df['orig_destination_distance'].notnull()]
df2['posa_continent'].unique()
df2['hotel_continent'].unique()

def cal_dist():
    datafr = pd.DataFrame(index = df2['posa_continent'].unique(), columns = df2['hotel_continent'].unique())
    datafr = datafr.fillna(0)
    for i in df2['posa_continent'].unique():
        for j in df2['hotel_continent'].unique():
            datafr.ix[i,j] = df2[(df2['posa_continent'] == i) & (df2['hotel_continent'] == j)]['orig_destination_distance'].mean()
    return datafr    

odd_df = cal_dist()

def impute_dist(df):
    cont1 = df['posa_continent']
    cont2 = df['hotel_continent']
    if pd.isnull(df['orig_destination_distance']):
        return odd_df.ix[cont1,cont2]
    else:
        return df['orig_destination_distance']

df['orig_destination_distance'] = df.apply(impute_dist, axis=1)

df.isnull().sum()

#Deleting 122 rows with missing search check in and out
df = df.drop(df[df['srch_ci'].isnull()].index, axis=0)
    
#-------------------------------------------------------------------------------------------

df1 = df[['user_id','is_booking']]
df1.groupby(['user_id','is_booking']).size().reset_index()\
.groupby(0)['is_booking'].agg({'total':'count', 'mean':'mean'})

df.groupby('is_mobile')['is_booking'].mean()

df.groupby('is_package')['is_booking'].mean()

df.groupby('channel')['is_booking'].agg({'Total':'count', 'Avg':'mean', 'Sum':'sum'})

#-------------------------------------------------------------------------------------------
#Adding features
#Insering total number of days between check in and checkout
for i in [i for i in df.columns if df[i].dtype == 'O'][:3]:
    df[i] = pd.to_datetime(df[i])

df.insert(13,'total_stay',df['srch_co'] - df['srch_ci'])

df['total_stay'] = df['total_stay'].apply(lambda x: str(x).split()[0])

#Difference between check in and booking date/search date
df.insert(14,'adv_booking',df['srch_ci'] - df['date_time'])

df['adv_booking'] = df['adv_booking'].apply(lambda x: str(x).split()[0])

#Splitting date column into multiple features
    
df.insert(1,'Day',df['date_time'].dt.dayofweek)

df.insert(2,'Month',df['date_time'].dt.month)

df.insert(3,'Year',df['date_time'].dt.year)

index = df.index
sample = pd.read_csv('sample.csv')

df['date_time'] = sample.ix[index, 'date_time']
df['date_time'] = pd.DatetimeIndex(df['date_time'])

df.insert(4,'Hour',df['date_time'].dt.hour)

df.insert(5,'Weekend',np.where((df['Day'] == 5) | (df['Day'] == 6),1,0))

df['total_stay'] = df['total_stay'].astype(int)
df['adv_booking'] = df['adv_booking'].astype(int)

df.insert(20,'days_dist',(df['total_stay']/df['orig_destination_distance']))

df.insert(21,'booking_dist',(df['adv_booking']/df['orig_destination_distance']))

df['ci_day'] = df['srch_ci'].dt.day

df.insert(19,'ci_mon',df['srch_co'].dt.month)

def hols_in_stay(df):
    tot_sat = 0 
    tot_sat = len(pd.date_range(df['srch_ci'], df['srch_co'], freq='W-SAT'))
    tot_sun = len(pd.date_range(df['srch_ci'], df['srch_co'], freq='W-SUN'))
    return tot_sat + tot_sun

df.insert(21, 'hols_in_stay',df.apply(hols_in_stay, axis=1))

df.insert(20, 'ci_dow',df['srch_ci'].dt.dayofweek)

df.insert(21, 'co_day',df['srch_co'].dt.day)

df.insert(22, 'co_dow',df['srch_co'].dt.dayofweek)

df.drop(['srch_ci','srch_co','date_time','user_id'], axis=1, inplace = True)

df.insert(21, 'hols_over_stay',df['hols_in_stay']/ df['total_stay'])

df['total_stay'] = df['total_stay']+1

df['hols_over_stay'] = df['hols_in_stay']/ df['total_stay']

df['days_dist'] = df['total_stay']/df['orig_destination_distance']

#--------------------------------------------------------------------------------------
#Outlier detection
#We have 5 continuous variables

plt.boxplot(df['orig_destination_distance'])

plt.boxplot(df['hols_over_stay'])

plt.boxplot(np.log(df['days_dist']))

#--------------------------------------------------------------------------------------

sns.heatmap(df.corr())
#SelectFromModel
select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
#select.fit(df.drop('is_booking', axis=1), df['is_booking'])
x_train, x_test, y_train, y_test = train_test_split(df.drop('is_booking', axis=1), df['is_booking'], test_size = 0.4, random_state=100)

select.fit(x_train,y_train)
df_select = select.transform(x_train)
x_test_s = select.transform(x_test)

#SelectKBest
skb = SelectKBest(f_classif,k=30)
skb.fit(x_train,y_train)
df.drop('is_booking', axis=1).columns[skb.get_support()]

x_train_s = skb.transform(x_train)
x_test_s =  skb.transform(x_test)

#RandomForest
clf = RandomForestClassifier(n_estimators=2000,max_depth=10,min_samples_split=5,bootstrap=True,max_features='sqrt',min_samples_leaf=1)
clf.fit(x_train_s,y_train)
pred = clf.predict(x_test_s)
pred1 = clf.predict(x_train_s)

print confusion_matrix(y_test, pred)
print confusion_matrix(y_train, pred1)

print classification_report(y_test, pred)

print classification_report(y_train, pred1)
    

#Adaboost
clf1 = AdaBoostClassifier(n_estimators=2000, learning_rate=1)
clf1.fit(x_train,y_train)
pred = clf1.predict(x_test)
pred1 = clf1.predict(x_train)


print confusion_matrix(y_test, pred)
print confusion_matrix(y_train, pred1)

print classification_report(y_test, pred)

print classification_report(y_train, pred1)

#xgboost
clf2 = xgb.XGBClassifier()
clf2.fit(x_train,y_train)
pred = clf2.predict(x_test)
pred1 = clf2.predict(x_train)

print confusion_matrix(y_test, pred)
print confusion_matrix(y_train, pred1)

print classification_report(y_test, pred)

print classification_report(y_train, pred1)

#-------------------------------------------------------------------------------------------------
#RandomizedSearch CV

clf2 = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=10)
clf2.fit(x_train_s,y_train)
pred = clf2.predict(x_test_s)
pred1 = clf2.predict(x_train_s)

print confusion_matrix(y_test, pred)
print confusion_matrix(y_train, pred1)

print classification_report(y_test, pred)
print classification_report(y_train, pred1)


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf.fit(x_train_s,y_train)

rf.best_params_

#-------------------------------------------------------------------------------------------------
#GridSearchCV

param_grid = {    
    'max_depth': [80, 100, 110],
    'min_child_weight': range(1,6,2),
    'gamma':[i/10.0 for i in range(0,5)],
    'learning_rate': [0.1,0.01,0.001,1],
    'n_estimators': [500, 1000, 1500],
    'subsample':[i/10.0 for i in range(6,10)]}

grid_search = GridSearchCV(estimator = clf2, param_grid = param_grid, cv = 5, n_jobs = -1, verbose = 2) 

grid_search.fit(x_train_s,y_train)

clf2.get_params().keys()