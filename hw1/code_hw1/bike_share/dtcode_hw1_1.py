import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# import to split the dataset, cross validation, Grid seach on cv
from sklearn.model_selection import train_test_split,GridSearchCV,cross_validate,learning_curve
# import metrics to calcuate loss
from sklearn.metrics import accuracy_score, confusion_matrix
# import to use decision tree
from sklearn.tree import DecisionTreeClassifier
# Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

def label_race(row):
    if row['count'] >= 145 :
        return 1
    else:
        return 0

def prep_clean_traindata(df):
    df['target'] = df.apply(lambda row: label_race(row), axis=1)
    df = df.drop(['count','datetime'], axis='columns')
    df['season'] = df['season'].astype('str')
    df['weather'] = df['weather'].astype('str')
    df['holiday'] = df['holiday'].astype('bool')
    df['workingday'] = df['workingday'].astype('bool')
    data0 = df.loc[df['target'] == 1][:1000]
    data1 = df.loc[df['target'] == 0][:1000]    
    result = pd.concat([data0, data1])
    return result

def prep_clean_testdata(df):
    df = df.drop(['datetime'], axis='columns')
    df['season'] = df['season'].astype('str')
    df['weather'] = df['weather'].astype('str')
    df['holiday'] = df['holiday'].astype('bool')
    df['workingday'] = df['workingday'].astype('bool')
    df = df[:1000]
    return df

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
    #               DECISION TREE                  #
    #                                              #
    ################################################
def decision_tree(X_train, y_train, parameters):
    '''
    This function is to build decision tree from Sklean
    Args:
        X_train --> features in training set which is a dataframe 
        y_train --> target in training set which is a dataframe 
        parameters --> parameters in decision tree

    '''
    ## find the best parameters
    dt = DecisionTreeClassifier(random_state = 42)
    clf = GridSearchCV(dt, parameters, scoring='balanced_accuracy', cv=5)
    model = clf.fit(X_train, y_train)
    return model.best_params_, model.best_score_



def plot_training_criterion(X_train, y_train, para):    
    train_sizes = [100, 300, 500, 700, 1120]
    para = para[0]
    dt_gini = DecisionTreeClassifier(criterion = 'gini', random_state = 42, max_depth = para['max_depth'], 
                                min_samples_leaf = para['min_samples_leaf'], min_impurity_decrease = para['min_impurity_decrease'], 
                                class_weight = 'balanced')
    dt_entropy = DecisionTreeClassifier(criterion = 'entropy', random_state = 42, max_depth = para['max_depth'], 
                                min_samples_leaf = para['min_samples_leaf'], min_impurity_decrease = para['min_impurity_decrease'], 
                                class_weight = 'balanced')
    train_sizes, train_scores_gini, validation_scores_gini = learning_curve(estimator = dt_gini,
                                                                              X = X_train,
                                                                              y = y_train, 
                                                                              train_sizes = train_sizes, 
                                                                              cv = 5,
                                                                              scoring = 'accuracy')

    train_sizes, train_scores_ent, validation_scores_ent = learning_curve(estimator = dt_entropy,
                                                                              X = X_train,
                                                                              y = y_train, 
                                                                              train_sizes = train_sizes, 
                                                                              cv = 5,
                                                                              scoring = 'accuracy')    
    
    train_scores_gini_mean = train_scores_gini.mean(axis = 1)
    train_scores_ent_mean = train_scores_ent.mean(axis = 1)
    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_gini_mean, label = 'Training gini accuracy')
    plt.plot(train_sizes, train_scores_ent_mean, label = 'Training entropy accuracy')
    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('Learning curves for a Decision Tree model', fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0,1)



def plot_training_learning(X_train, y_train, para):    
    train_sizes = [100, 300, 500, 700, 1120]
    para = para[0]
    dt = DecisionTreeClassifier(criterion = para['criterion'], random_state = 42, max_depth = para['max_depth'], 
                                min_samples_leaf = para['min_samples_leaf'], min_impurity_decrease = para['min_impurity_decrease'], 
                                class_weight = 'balanced')
    train_sizes, train_scores, validation_scores = learning_curve(estimator = dt,
                                                                  X = X_train,
                                                                  y = y_train, 
                                                                  train_sizes = train_sizes, 
                                                                  cv = 5,
                                                                  scoring = 'accuracy')
    
    train_scores_mean = train_scores.mean(axis = 1)
    validation_scores_mean = validation_scores.mean(axis = 1)
    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label = 'Training accuracy')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation accuracy')
    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    plt.title('Learning curves for a Decision Tree model', fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0,1)


def plot_training_test(X_test, y_test, para):    
    train_sizes = [100, 200, 300, 480]
    para = para[0]
    dt = DecisionTreeClassifier(criterion = para['criterion'], random_state = 42, max_depth = para['max_depth'], 
                                min_samples_leaf = para['min_samples_leaf'], min_impurity_decrease = para['min_impurity_decrease'], 
                                class_weight = 'balanced')
    train_sizes, test_scores, validation_scores = learning_curve(estimator = dt,
                                                                  X = X_test,
                                                                  y = y_test, 
                                                                  train_sizes = train_sizes, 
                                                                  cv = 5,
                                                                  scoring = 'accuracy')
    
    test_scores_mean = test_scores.mean(axis = 1)
    plt.style.use('seaborn')
    plt.plot(train_sizes, test_scores_mean, label = 'Testing accuracy')
    plt.ylabel('Accuracy', fontsize = 14)
    plt.xlabel('Testing set size', fontsize = 14)
    plt.title('Learning curves for a Decision Tree model', fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0,1)

    ################################################
    #                                              #
    #            END OF DECISION TREE              #
    #                                              #
    ################################################



def main():
    # read two csv files 
    train = pd.read_csv('./bike-sharing-demand/train.csv', header = 0, sep = ",")
    test = pd.read_csv('./bike-sharing-demand/test.csv', header = 0, sep = ",")

    # clean and prep tables
    train_df = prep_clean_traindata(train)
    test_df = prep_clean_testdata(test)

    # prep training and testing
    X = train_df.drop(['target'], axis='columns')
    y = train_df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    ################################################
    #                                              #
    #               DECISION TREE                  #
    #                                              #
    ################################################
    # set grid search parameters
    parameters = {'criterion': ['gini','entropy'],
                'max_depth': list(range(3,15)),
                'min_samples_leaf': [50,100,125],
                'min_impurity_decrease':[0.001,0.01],
                'class_weight':['balanced']}

    # find the best parameter and its model results
    para = decision_tree(X_train, y_train, parameters)

    # plot different graphs
    plot_training_criterion(X_train, y_train, para)
    plot_training_learning(X_train, y_train, para)
    plot_training_test(X_test, y_test, para)

    max_depth = []
    acc_entropy_train = []
    acc_entropy_test = []
    para = para[0]
    for i in range(1,15):
        dt_entropy = DecisionTreeClassifier(criterion = 'entropy', random_state = 42, max_depth = i, 
                                    min_samples_leaf = para['min_samples_leaf'], min_impurity_decrease = para['min_impurity_decrease'], 
                                    class_weight = 'balanced')
        model = dt_entropy.fit(X_train, y_train)
        pred1 = model.predict(X_train)
        pred2 = model.predict(X_test)
        acc_entropy_train.append(accuracy_score(y_train, pred1))
        acc_entropy_test.append(accuracy_score(y_test, pred2))
        max_depth.append(i)
        
    d = pd.DataFrame({'acc_entropy_train':pd.Series(acc_entropy_train), 'acc_entropy_test':pd.Series(acc_entropy_test), 'max_depth':pd.Series(max_depth)})
    # visualizing changes in parameters
    plt.plot('max_depth','acc_entropy_train', data=d, label='entropy_train')
    plt.plot('max_depth','acc_entropy_test', data=d, label='entropy_test')
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    plt.legend()
    ################################################
    #                                              #
    #            END OF DECISION TREE              #
    #                                              #
    ################################################

    if __name__ == '__main__':
        main()