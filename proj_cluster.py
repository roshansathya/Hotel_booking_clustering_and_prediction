#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 18:14:29 2018

@author: roshan
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import pylab as pl
from scipy import stats
from sklearn.externals.six import StringIO
from sklearn import preprocessing
from sklearn import cluster,tree, decomposition
import matplotlib.pyplot as plt
import pydot

df = pd.read_csv('sample.csv')

df.head()
df['user_location_city'].nunique()
df['user_location_region'].nunique()
df.isnull().sum()
df.info()

pd.crosstab(df['is_booking'],df['srch_rm_cnt'])

pd.crosstab(df['is_booking'],df['channel'])

df.groupby('srch_rm_cnt')['is_booking'].mean()

df[df['srch_rm_cnt'] == 0]

##Deleting row with no room count
df = df.drop(7821, axis=0)

df['srch_children_cnt'].corr(df['is_booking'])
corr = df.corr()

df[['channel', 'is_booking', 'is_mobile', 'orig_destination_distance', 'srch_rm_cnt', 'srch_adults_cnt', 'srch_children_cnt']].hist()

df['user_id'].nunique()

(df.groupby('user_id')['is_booking']\
   .agg(['mean']).reset_index()\
   .groupby('mean')['user_id']\
   .agg('count').reset_index()).plot(x='mean', y='user_id')

pd.crosstab(df['srch_adults_cnt'], df['srch_children_cnt'])

#Deleting rows where adult and child = 0
df = df.drop(df[df['srch_adults_cnt'] + df['srch_children_cnt'] == 0].index, axis=0)

#Convering datetime columns to datetime objects
for i in [i for i in df.columns if df[i].dtype == 'O']:
    df[i] = pd.to_datetime(df[i])
    
for i in [i for i in df.columns if df[i].dtype == 'M8[ns]']:
    df[i] = df[i].apply(lambda x:x.date())

#-----------------------------------------------------------------------------------------------------

#Checking for 3 business logics:
# 1. Booking date < check in date
# 2. Check in date < Check out date
# 3. Adults = 0 and child > 0   
    
df = df.drop((df[df["date_time"] > df['srch_ci']]).index,axis=0)

df = df.drop((df[df["srch_ci"] > df['srch_co']]).index,axis=0)

df[(df['srch_adults_cnt'] == 0) & (df['srch_children_cnt'] > 0)][['srch_adults_cnt','srch_children_cnt','orig_destination_distance']]

df = df.drop((df[(df['srch_adults_cnt'] == 0) & (df['srch_children_cnt'] > 0)]).index,axis = 0)

df.drop('Unnamed: 0', axis=1, inplace = True)

#------------------------------------------------------------------------------------------------------
#Insering total number of days between check in and checkout
df.insert(13,'total_stay',df['srch_co'] - df['srch_ci'])

df['total_stay'] = df['total_stay'].apply(lambda x: str(x).split()[0])

df.insert(14,'adv_booking',df['srch_ci'] - df['date_time'])

df['adv_booking'] = df['adv_booking'].apply(lambda x: str(x).split()[0])

channel_perf = df.groupby('channel')['is_booking'].agg({'booking_rate':'mean', 'num_of_bookings':'sum', 'total':'size'})\
.reset_index().sort_values('booking_rate')

stats.ttest_ind(channel_perf['total'], channel_perf['booking_rate'])

num_list = ['total_stay', 'adv_booking', 'orig_destination_distance', 'is_mobile',
            'is_package', 'srch_adults_cnt', 'srch_children_cnt', 'srch_rm_cnt', 'user_location_city']

city_data = df.dropna(axis=0)[num_list]

city_groups = city_data.groupby('user_location_city').mean().reset_index().dropna(axis=0)

sc = preprocessing.StandardScaler()

sc.fit(city_groups)
city_gps_s = sc.transform(city_groups)

km = cluster.KMeans(n_clusters=3,max_iter=300, random_state=None)

city_groups['cluster'] = km.fit_predict(city_gps_s)   

#-----------------------------------------------------------------------------------------------------
#Using PCA to reduce dimensionality to find clusters
pca = decomposition.PCA(n_components=2, whiten=True)
pca.fit(city_groups[num_list])

plt.scatter(city_groups['x'], city_groups['y'], c=city_groups['cluster'])
plt.show()

city_groups.groupby(['cluster']).mean()
