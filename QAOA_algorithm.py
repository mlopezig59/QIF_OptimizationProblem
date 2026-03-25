import numpy as np
import pandas as pd
import math
from datetime import datetime

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.optimizers import COBYLA

# ------------------------
# Distance
# ------------------------
def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# ------------------------
# Construire W (r -> s)
# ------------------------
def build_weight_matrix(df, threshold):
    n = len(df)
    W = np.zeros((n, n))

    for r in range(n):
        for s in range(n):
            if r != s:
                d = distance(
                    df.iloc[r]["latitude"], df.iloc[r]["longitude"],
                    df.iloc[s]["latitude"], df.iloc[s]["longitude"]
                )
                if d < threshold:
                    W[r, s] = df.iloc[r]["population_est"] / (1 + d)

    return W

# ------------------------
# Construire coefficients c_s = sum_r w_rs
# ------------------------
def compute_source_weights(W):
    return np.sum(W, axis=0)

# ------------------------
# Construire QUBO
# ------------------------
def build_qubo(c):
    n = len(c)
    qp = QuadraticProgram()

    # variables binaires x_s
    for i in range(n):
        qp.binary_var(name=f"x_{i}")

    # linear objective
    linear = {f"x_{i}": float(c[i]) for i in range(n)}

    qp.maximize(linear=linear)

    return qp

# MAIN
df = pd.read_csv("main_cities.csv")

# parameters
DISTANCE_THRESHOLD = 0.4

# construction
W = build_weight_matrix(df, DISTANCE_THRESHOLD)
c = compute_source_weights(W)

qp = build_qubo(c)

# QAOA
def callback(eval_count, params, value, metadata):
    print(f"Iteration {eval_count} - value = {value}")

qaoa = QAOA(sampler=StatevectorSampler(seed=123), optimizer=COBYLA(maxiter=30), callback = callback)

solver = MinimumEigenOptimizer(qaoa)

print("Début solve...")
start_time = datetime.now()
result = solver.solve(qp)
end_time = datetime.now()
print(f"Time : {end_time-start_time} s")
print("Solve terminé !")

# ------------------------
# Résultats
# ------------------------
print("Solution binaire :", result.x)
print("Valeur objectif :", result.fval)