import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# import to split the dataset, cross validation, Grid seach on cv
from sklearn.model_selection import train_test_split,GridSearchCV,cross_validate,learning_curve
# import metrics to calcuate loss
from sklearn.metrics import accuracy_score, confusion_matrix
# import to use boosting
from sklearn.ensemble import AdaBoostClassifier
# Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

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

def visualizing(df, column, target, types):
    '''
    This function is to plot features
    Args:
        df --> fdataframe 
        column --> one feature in dataset
        target --> target feature in dataset 
        types --> feature type such as numeric or others
        
    '''  
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('distribution and correlation')
    if types == "numeric":
        df[column].plot(kind = 'box', ax = ax1)
        df.plot(kind ='scatter',x = target, y = column, color = 'red', ax = ax2)
    else:
        df[column].plot(kind = 'hist', ax = ax1)
        df[target].plot(kind = 'hist', ax = ax2)


    ################################################
    #                                              #
    #                  Adaboost                    #
    #                                              #
    ################################################
def boosting(X_train, y_train, parameters):
    ## find the best parameters
    bt = AdaBoostClassifier(random_state=42)
    clf = GridSearchCV(bt, parameters, scoring='balanced_accuracy', cv=5, n_jobs=-1)
    model = clf.fit(X_train, y_train)
    return model.best_params_, model.best_score_

def plot_training_algo(X_train, y_train, para):    
    train_sizes = [1000, 20000, 40000, 60000, 94143]
    para = para[0]
    boost_sr = AdaBoostClassifier(algorithm = 'SAMME.R', learning_rate = para['learning_rate'], n_estimators = para['n_estimators'], random_state=42)
    
    boost_se = AdaBoostClassifier(algorithm = 'SAMME', learning_rate = para['learning_rate'], n_estimators = para['n_estimators'], random_state=42)
    train_sizes, train_scores_sr, validation_scores_sr = learning_curve(estimator = boost_sr,
                                                                              X = X_train,
                                                                              y = y_train, 
                                                                              train_sizes = train_sizes, 
                                                                              cv = 5,
                                                                              scoring = 'balanced_accuracy')

    train_sizes, train_scores_se, validation_scores_se = learning_curve(estimator = boost_se,
                                                                              X = X_train,
                                                                              y = y_train, 
                                                                              train_sizes = train_sizes, 
                                                                              cv = 5,
                                                                              scoring = 'balanced_accuracy')    
    
    train_scores_se_mean = train_scores_se.mean(axis = 1)
    train_scores_sr_mean = train_scores_sr.mean(axis = 1)
    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_sr_mean, label = 'Training sr accuracy')
    plt.plot(train_sizes, train_scores_se_mean, label = 'Training se accuracy')
    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('Learning curves for a Adaboost model', fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0,1)

def plot_training_bstlearning(X_train, y_train, para):    
    train_sizes = [1000, 20000, 40000, 60000, 94143]
    para = para[0]
    boost_sr = AdaBoostClassifier(algorithm = para['algorithm'], learning_rate = para['learning_rate'], n_estimators = para['n_estimators'], random_state=42)
    train_sizes, train_scores_sr, validation_scores_sr = learning_curve(estimator = boost_sr,
                                                                              X = X_train,
                                                                              y = y_train, 
                                                                              train_sizes = train_sizes, 
                                                                              cv = 5,
                                                                              scoring = 'balanced_accuracy')
    
    train_scores_sr_mean = train_scores_sr.mean(axis = 1)
    validation_scores_sr_mean = validation_scores_sr.mean(axis = 1)
    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_sr_mean, label = 'Training accuracy')
    plt.plot(train_sizes, validation_scores_sr_mean, label = 'Validation accuracy')
    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('Learning curves for Adaboost model', fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0,1)

def plot_test_bst(X_test, y_test, para):    
    train_sizes = [1000, 20000, 40000, 40348]
    para = para[0]
    boost_sr = AdaBoostClassifier(algorithm = para['algorithm'], learning_rate = para['learning_rate'], n_estimators = para['n_estimators'], random_state=42)
    train_sizes, test_scores, validation_scores = learning_curve(estimator = boost_sr,
                                                                          X = X_test,
                                                                          y = y_test, 
                                                                          train_sizes = train_sizes, 
                                                                          cv = 5,
                                                                          scoring = 'balanced_accuracy')
    
    test_scores_mean = test_scores.mean(axis = 1)
    plt.style.use('seaborn')
    plt.plot(train_sizes, test_scores_mean, label = 'Testing accuracy')
    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Testing set size', fontsize = 14)
    plt.title('Learning curves for a knn model', fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0,1)
    ################################################
    #                                              #
    #                END OF Adaboost               #
    #                                              #
    ################################################

def main():
    # read two csv files 
    edges = pd.read_csv('./twitch_gamers/large_twitch_edges.csv', header = 0, sep = ",")
    features = pd.read_csv('./twitch_gamers/large_twitch_features.csv', header = 0, sep = ",")

    # clean and prep tables
    df1 = prep_data(edges, features)
    df2 = preprocessing(df1)

    # prep training and testing
    X = df2.drop(['dead_account'], axis='columns')
    y = df2['dead_account']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    ################################################
    #                                              #
    #               DECISION TREE                  #
    #                                              #
    ################################################
    # set grid search parameters
    parameters = {'n_estimators': [40, 50, 60, 70, 80],
                'learning_rate': [0.75, 1, 1.25],
                'algorithm': ['SAMME','SAMME.R']}

    para = boosting(X_train, y_train, parameters)
    plot_training_algo(X_train, y_train, para)
    plot_training_bstlearning(X_train, y_train, para)
    plot_test_bst(X_test, y_test, para)
    learning_rate = []
    acc_train = []
    acc_test = []
    para = para[0]
    for i in [0.25, 0.5, 0.75, 1]:
        boost_sr = AdaBoostClassifier(algorithm = para['algorithm'], learning_rate = i, n_estimators = para['n_estimators'], random_state=42)
        model = boost_sr.fit(X_train, y_train)
        pred1 = model.predict(X_train)
        pred2 = model.predict(X_test)
        acc_train.append(balanced_accuracy_score(y_train, pred1))
        acc_test.append(balanced_accuracy_score(y_test, pred2))
        learning_rate.append(i)
        
    d = pd.DataFrame({'acc_train':pd.Series(acc_train), 'acc_test':pd.Series(acc_test), 'learning_rate':pd.Series(learning_rate)})
    # visualizing changes in parameters
    plt.plot('learning_rate','acc_train', data=d, label='train')
    plt.plot('learning_rate','acc_test', data=d, label='test')
    plt.xlabel('learning_rate')
    plt.ylabel('accuracy')
    plt.legend()
    ################################################
    #                                              #
    #                END OF Adaboost               #
    #                                              #
    ################################################

    if __name__ == '__main__':
        main()