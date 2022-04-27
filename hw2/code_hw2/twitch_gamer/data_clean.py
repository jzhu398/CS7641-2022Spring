import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prep_data(edges, features):
    '''
    This function is to help join create features and join table
    Args:
        edges --> dataframe from read edges.csv
        features --> dataframe from read features.csv

    '''
    # find the mutual friends from edge and edge_inverse table
    # create edges inverse table
    edges_inverse = edges[['numeric_id_2','numeric_id_1']]
    # union them all together
    union = pd.concat([edges, edges_inverse], ignore_index=True)
    # Group by the key and count the mutual friends
    friends = pd.DataFrame({'mutual_friend' : union.groupby(['numeric_id_1'])['numeric_id_2'].count()}).reset_index()
    # drop na
    features = features.dropna()
    # join friends and features table together
    tmp_table = pd.merge(features, friends, left_on = 'numeric_id', right_on = 'numeric_id_1', how = 'left').drop(['numeric_id_1','created_at','updated_at'], axis='columns')
    # create target - which is churned customer
    tmp_table = tmp_table.drop(['mature'], axis='columns')
    return tmp_table

def preprocessing(df):
    '''
    This function is to help preprocessing data such as cleaning, changing data type, etc.
    Args:
        df --> dataframe

    '''
    # list column names
    column_name = df.columns
    # replace na with 0
    if df.isna().sum().sum() != 0:
        non_zero_df = df.fillna(0)
    # change to right data type
    non_zero_df['affiliate'] = non_zero_df['affiliate'].astype('bool')
    non_zero_df['mutual_friend'] = non_zero_df['mutual_friend'].astype('int64')
    # outliers - data may not have outliers then get rid of this step
    # scale
    new_list = []
    list_ob = []
    for names in column_name:
        if non_zero_df[names].dtypes == 'int64':
            new_list.append(names)
        if non_zero_df[names].dtypes == 'object':
            list_ob.append(names)
    scaler = MinMaxScaler()
    non_zero_df[new_list] = scaler.fit_transform(non_zero_df[new_list])
    # one-hot encoding
    for val in list_ob:
        df_dum = pd.get_dummies(non_zero_df[val], prefix=val)
        non_zero_df = pd.concat([non_zero_df, df_dum],axis=1).drop([val], axis='columns')
        
    non_zero_df['dead_account'] = non_zero_df['dead_account'].astype('int64').astype('str')
    return non_zero_df

def balanced_data(df, target, num):
    data0 = df.loc[df[target] == '0'][0:num]
    data1 = df.loc[df[target] == '1'][0:num]
    result = pd.concat([data0, data1])
    return result