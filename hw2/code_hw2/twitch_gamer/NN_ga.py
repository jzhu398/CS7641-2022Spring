import data_clean
import mlrose_hiive
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns

def results_to_plot(X_train_array, X_test_array, y_train_hot, y_test_hot, algorithms, layers):
    results = []
    for i in range(0, 5600, 200):
        start = time.time()
        model = mlrose_hiive.NeuralNetwork(hidden_nodes=[layers], activation='relu',
                                        algorithm=algorithms, max_iters=i,
                                        bias=True, is_classifier=True, learning_rate=0.1,
                                        early_stopping=True, clip_max=5, max_attempts=100,
                                        random_state=614)
        model.fit(X_train_array, y_train_hot)
        y_train_pred = model.predict(X_train_array)
        y_train_accuracy = accuracy_score(y_train_hot, y_train_pred)

        y_test_pred = model.predict(X_test_array)
        y_test_accuracy = accuracy_score(y_test_hot, y_test_pred)

        f1score = f1_score(y_test_hot, y_test_pred, average='weighted')
        end = time.time()
        results.append([i, algorithms, layers, y_train_accuracy, y_test_accuracy, f1score, round((end - start),3)])
    return results

def main():
    # read two csv files 
    edges = pd.read_csv('./twitch_gamers/large_twitch_edges.csv', header = 0, sep = ",")
    features = pd.read_csv('./twitch_gamers/large_twitch_features.csv', header = 0, sep = ",")

    # clean and prep tables
    df1 = data_clean.prep_data(edges, features)
    df2 = data_clean.preprocessing(df1)
    df3 = data_clean.balanced_data(df2, 'dead_account', 5000)

    # prep training and testing
    X = df3.drop(['dead_account'], axis='columns')
    y = df3['dead_account']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    #one hot encoding the target variables
    one_hot = OneHotEncoder()
    y_train_hot = one_hot.fit_transform(y_train.values.reshape(-1, 1)).todense()
    y_test_hot = one_hot.transform(y_test.values.reshape(-1, 1)).todense()
    X_train_array = X_train.to_numpy().astype(float)
    X_test_array = X_test.to_numpy().astype(float)

    algorithms = ['random_hill_climb', 'simulated_annealing', 'genetic_alg']
    layers = [3,6,9,12]

    result_3_rhc = results_to_plot(X_train_array, X_test_array, y_train_hot, y_test_hot, algorithms[2], layers[0])
    result_6_rhc = results_to_plot(X_train_array, X_test_array, y_train_hot, y_test_hot, algorithms[2], layers[1])
    result_9_rhc = results_to_plot(X_train_array, X_test_array, y_train_hot, y_test_hot, algorithms[2], layers[2])
    result_12_rhc = results_to_plot(X_train_array, X_test_array, y_train_hot, y_test_hot, algorithms[2], layers[3])

    df_3_rhc = pd.DataFrame(result_3_rhc, columns=["Iterations", "Algorithm", "layers", "Train_Accuracy", "Test_Accuracy", "F1_Score", "running_time"])
    df_6_rhc = pd.DataFrame(result_6_rhc, columns=["Iterations", "Algorithm", "layers", "Train_Accuracy", "Test_Accuracy", "F1_Score", "running_time"])
    df_9_rhc = pd.DataFrame(result_9_rhc, columns=["Iterations", "Algorithm", "layers", "Train_Accuracy", "Test_Accuracy", "F1_Score", "running_time"])
    df_12_rhc = pd.DataFrame(result_12_rhc, columns=["Iterations", "Algorithm", "layers", "Train_Accuracy", "Test_Accuracy", "F1_Score", "running_time"])
    result_rhc = pd.concat([df_3_rhc, df_6_rhc, df_9_rhc, df_12_rhc])

    result_rhc_1 = result_rhc.pivot("Iterations", "layers", "Train_Accuracy")
    result_rhc_2 = result_rhc.pivot("Iterations", "layers", "Test_Accuracy")
    result_rhc_3 = result_rhc.pivot("Iterations", "layers", "running_time")
    result_rhc_4 = result_rhc.pivot("Iterations", "layers", "F1_Score")

    sns.lineplot(data=result_rhc_1)
    sns.lineplot(data=result_rhc_2)
    sns.lineplot(data=result_rhc_3)
    sns.lineplot(data=result_rhc_4)

    best = []
    for i in range(30,150,30):
        model = mlrose_hiive.NeuralNetwork(hidden_nodes=[3], activation='relu',
                                        algorithm='genetic_alg', max_iters= 300,
                                        bias=True, is_classifier=True, learning_rate=0.1,
                                        early_stopping=True, clip_max=5, max_attempts=100, 
                                        pop_size = i, 
                                        random_state=614)
        model.fit(X_train_array, y_train_hot)
        y_test_pred = model.predict(X_test_array)
        f1score = f1_score(y_test_hot, y_test_pred, average='weighted')
        best.append([i, f1score])
    df_best = pd.DataFrame(best, columns=["Iterations", "F1_Score"])
    sns.lineplot(data=df_best, x="Iterations", y="F1_Score")

    best_1 = []
    for i in range(1,5,1):
        model = mlrose_hiive.NeuralNetwork(hidden_nodes=[3], activation='relu',
                                        algorithm='genetic_alg', max_iters= 300,
                                        bias=True, is_classifier=True, learning_rate=0.1,
                                        early_stopping=True, clip_max=5, max_attempts=100, 
                                        pop_size = 90, 
                                        mutation_prob = (i/4),
                                        random_state=614)
        model.fit(X_train_array, y_train_hot)
        y_test_pred = model.predict(X_test_array)
        f1score = f1_score(y_test_hot, y_test_pred, average='weighted')
        best_1.append([i, f1score])
    df_best_1 = pd.DataFrame(best_1, columns=["Iterations", "F1_Score"])
    sns.lineplot(data=df_best_1, x="Iterations", y="F1_Score")