import mlrose_hiive
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

def rhc(problem_fit, problem_name, max_iters= 100):
    start = time.time()
    fitness_score = mlrose_hiive.random_hill_climb(problem_fit, max_attempts=100, max_iters= max_iters, restarts=10, random_state = 614)[1]
    return [max_iters, "random_hill_climb", problem_name,fitness_score, time.time()-start]

def sa(problem_fit, problem_name, max_iters= 100):
    start = time.time()
    fitness_score = mlrose_hiive.simulated_annealing(problem_fit, max_attempts=100, max_iters= max_iters, random_state = 614)[1]
    return [max_iters, "simulated_annealing", problem_name,fitness_score, time.time()-start]

def ga(problem_fit, problem_name, max_iters= 100):
    start = time.time()
    fitness_score = mlrose_hiive.genetic_alg(problem_fit, max_attempts=100, max_iters= max_iters, pop_size= 200, mutation_prob=0.1, random_state = 614)[1]
    return [max_iters, "genetic_alg", problem_name,fitness_score, time.time()-start]

def mimic(problem_fit, problem_name, max_iters= 100):
    start = time.time()
    fitness_score = mlrose_hiive.mimic(problem_fit, pop_size=200, keep_pct=0.2, max_attempts=10, max_iters=max_iters, curve=False, random_state=614)[1]
    return [max_iters, "mimic", problem_name,fitness_score, time.time()-start]


results = []
edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
# Queens - simulated annealing
# Continous Peaks - genetic algorithm
# One Max - MIMIC
problems_name = ["Queens", "Continuous Peaks", "One Max"]
fitness_functions = [mlrose_hiive.Queens(), mlrose_hiive.ContinuousPeaks(), mlrose_hiive.OneMax()]
problems = [mlrose_hiive.DiscreteOpt(length = 100, fitness_fn = fitness_function, maximize=True, max_val = 2) for fitness_function in fitness_functions]

for j in range(len(problems)):
    for i in range(0, 3000, 100):
        results.append(rhc(problems[j], problems_name[j], max_iters= i))
        results.append(sa(problems[j], problems_name[j], max_iters= i))
        results.append(ga(problems[j], problems_name[j], max_iters= i))
        results.append(mimic(problems[j], problems_name[j], max_iters= i))

df = pd.DataFrame(results, columns=["Iteration", "Algorithm", "Problem","Fitness", "Time"])
for problem in problems_name:
  plt.figure()
  sns.lineplot(data=df[df['Problem']==problem], x="Iteration", y="Fitness", hue="Algorithm").set_title(problem+ ": Fitness vs Iterations")
  plt.figure()
  sns.lineplot(data=df[df['Problem']==problem], x="Iteration", y="Time", hue="Algorithm").set_title(problem+  ": Time vs Iterations")

print(df.groupby(['Algorithm', 'Problem'])['Fitness'].max())
print(df.groupby(['Algorithm', 'Problem'])['Time'].max())
print(df[df['Problem']=='Queens'].groupby(['Algorithm', 'Problem'])['Fitness'].mean())
print(df[df['Problem']=='Queens'].groupby(['Algorithm', 'Problem'])['Time'].mean())

print(df[df['Problem']=="Continuous Peaks"].groupby(['Algorithm', 'Problem'])['Fitness'].mean())
print(df[df['Problem']=="Continuous Peaks"].groupby(['Algorithm', 'Problem'])['Time'].mean())

print(df[df['Problem']=="One Max"].groupby(['Algorithm', 'Problem'])['Fitness'].mean())
print(df[df['Problem']=="One Max"].groupby(['Algorithm', 'Problem'])['Time'].mean())

sns.lineplot(data=df[(df['Problem']=="One Max") & (df['Iteration'] < 200)], x="Iteration", y="Fitness", hue="Algorithm").set_title(problem+ ": Fitness vs Iterations")
sns.lineplot(data=df[(df['Problem']=='Max-K Color') & (df['Iteration'] < 200)], x="Iteration", y="Time", hue="Algorithm").set_title(problem+ ": Time vs Iterations")