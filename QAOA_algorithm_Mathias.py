import numpy as np
import pandas as pd
import math
from datetime import datetime

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import COBYLA

# Distance
def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# Build weight matrix W
def build_weight_matrix(df, threshold, n):
    W = np.zeros((n, n))

    for r in range(n):
        for s in range(n):
            if r != s:
                d = distance(
                    df.iloc[r]["latitude"], df.iloc[r]["longitude"],
                    df.iloc[s]["latitude"], df.iloc[s]["longitude"]
                )
                if d < threshold:
                    W[r, s] = 1

    return W

# Build coefficients c_s = sum_r w_rs
def compute_source_weights(W):
    return np.sum(W, axis=0)

# Build QUBO
def build_qubo(c, mu):
    n = len(c)
    qp = QuadraticProgram()

    # binary variables x_s , 
    for i in range(n):
        qp.binary_var(name=f"x_{i}")

    # linear objective
    linear = {f"x_{i}": float(c[i]) for i in range(n)}

    qp.maximize(linear=linear)

    return qp


df = pd.read_csv("main_cities.csv")

# parameters
DISTANCE_THRESHOLD = 0.4
city_number = 6
mu = 1.1

W = build_weight_matrix(df, DISTANCE_THRESHOLD, city_number)
c = compute_source_weights(W)

qp = build_qubo(c, mu)

# QAOA
def callback(eval_count, params, value, metadata):
    print(f"Iteration {eval_count} - value = {value}")

qaoa = QAOA(sampler=StatevectorSampler(seed=123), optimizer=COBYLA(maxiter=1000), reps=10, callback = callback)

solver = MinimumEigenOptimizer(qaoa)

print("Start solve...")
start_time = datetime.now()
result = solver.solve(qp)
end_time = datetime.now()
print(f"Time : {end_time-start_time} s")
print("Solve finished !")

# Results
print("Binary solution :", result.x)
print("Target value :", result.fval)