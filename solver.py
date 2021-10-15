import json
from math import *
import matplotlib.pyplot as plt
import csv
from sympy import *
import numpy as np
from numpy.linalg import inv

# Неявная схема
with open("task.json", 'r') as task_:
    task = json.load(task_)

# задание функций , описывающих правую часть и начальные условия
t = Symbol("t")
x = Symbol("x")
ICS_parsed = parse_expr(task["ICS"])
f_parsed = parse_expr(task["rightPart"])
ICS = lambdify(x, ICS_parsed)
f = lambdify([t,x], f_parsed)

# задание параметров сетки
h = task["solver"]["h"]
tau = task["solver"]["tau"]
a = task["a"]
x1 = task["x1"]
x0 = task["x0"]
n = int((x1-x0)/h)
n_t = 801

# формирование матрицы СЛАУ
beta0 = task["BC"][0]
beta1 = task["BC"][1]
p = -(a**2) * tau / (h ** 2)
q = 1 + 2 * (a**2) * tau / (h ** 2)
r = p
H = np.zeros((n + 1, n + 1))

for i in range(1, H.shape[0] - 1):
    H[i, i-1] = p
    H[i, i] = q
    H[i, i+1] = r
H[0, 0] = 1
H[n, n] = 1

U_t_x = np.zeros((n_t , n + 1))

# задание начальных условий
U_t_x[0] = np.array([ICS(k*h) for k in range(n + 1)])

#решение уравнения теплопроводности
t_i = 0
for j in range(1, n_t ):
    b_0_vec = np.zeros((n + 1, 1))
    b_1_vec = np.zeros((n + 1, 1))
    f_vec = np.zeros((n + 1, 1))
    x_i = 0
    for i in range(1, n):
        f_vec[i, 0] = tau * f(t_i, x_i)
        x_i += h
    t_i += tau
    b_0_vec[0, 0] = beta0
    b_0_vec[n, 0] = beta1
    b_1_vec[0, 0] = beta0
    b_1_vec[n, 0] = beta1
    U_t_x[j] = (np.dot(inv(H),((U_t_x[j - 1]).reshape((n + 1, 1)) - b_1_vec + b_0_vec + f_vec))).reshape((n + 1))


u_print = U_t_x[0]
plt.plot(np.linspace(0,1,n+1),u_print)
plt.show()
