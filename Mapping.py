import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler
from scipy.optimize import minimize
from qiskit.quantum_info import Statevector


# ------------------------
# Distance
# ------------------------
def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

# ------------------------
# Construire W
# ------------------------
def build_weight_matrix(df, threshold):
    n = len(df)
    W = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                d = distance(
                    df.iloc[i]["latitude"], df.iloc[i]["longitude"],
                    df.iloc[j]["latitude"], df.iloc[j]["longitude"]
                )
                if d < threshold:
                    W[i, j] = df.iloc[j]["population_est"] / (1 + d)
    return W

# ------------------------
# Construire H_C (Pauli Z)
# ------------------------
def build_cost_hamiltonian(W):
    n = W.shape[0]
    paulis = []
    coeffs = []

    # constante (optionnelle)
    constant = 0

    for i in range(n):
        for j in range(i+1, n):
            if W[i, j] != 0:
                w = W[i, j]

                z = ['I'] * n
                z[i] = 'Z'
                z[j] = 'Z'
                paulis.append("".join(z))
                coeffs.append(w / 4)

                z = ['I'] * n
                z[i] = 'Z'
                paulis.append("".join(z))
                coeffs.append(-w / 4)

                z = ['I'] * n
                z[j] = 'Z'
                paulis.append("".join(z))
                coeffs.append(-w / 4)

                constant += w / 4

    H = SparsePauliOp(paulis, coeffs)

    return H, constant

# ------------------------
# Circuit QAOA
# ------------------------
def qaoa_circuit(gamma, beta, Hc, n):
    qc = QuantumCircuit(n)

    # Etat initial |+>
    qc.h(range(n))

    # ----- COST -----
    for pauli, coeff in zip(Hc.paulis, Hc.coeffs):
        indices = [i for i, p in enumerate(pauli.to_label()) if p == 'Z']
        if len(indices) == 2:
            i, j = indices

            qc.cx(i, j)
            qc.rz(2 * gamma * float(np.real(coeff)), j)
            qc.cx(i, j)

    # ----- MIXER -----
    for i in range(n):
        qc.rx(2 * beta, i)

    return qc

# ------------------------
# Fonction objectif
# ------------------------
def expectation(params, Hc, n):
    gamma, beta = params

    qc = qaoa_circuit(gamma, beta, Hc, n)

    state = Statevector.from_instruction(qc)

    # calcul <psi|H|psi>
    value = np.real(state.expectation_value(Hc)) #+ constant
    return -value  # minimisation

# ------------------------
# MAIN
# ------------------------
df = pd.read_csv("main_cities.csv")
n = len(df)

W = build_weight_matrix(df, threshold=0.4)
Hc, constant = build_cost_hamiltonian(W)

# Optimisation classique
init = [0.1, 0.1]

res = minimize(
    expectation,
    init,
    args=(Hc, n),
    method="COBYLA"
)

print("Paramètres optimaux :", res.x)

# ------------------------
# Solution finale
# ------------------------
qc_final = qaoa_circuit(res.x[0], res.x[1], Hc, n)

state = Statevector.from_instruction(qc_final)
probs = state.probabilities_dict()

# meilleure solution
best_state = max(probs, key=probs.get)

print("Meilleure solution binaire :", best_state)

# plt.figure()
# plt.scatter(latitudes, longitudes)