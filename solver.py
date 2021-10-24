import json
from math import *
import matplotlib.pyplot as plt
import csv
from sympy import *
import numpy as np
from numpy.linalg import inv

# Рассматривается неявная схема решения задачи теплопроводности с ГУ 1го рода
with open("task.json", 'r') as task_:
    task = json.load(task_)

# задание функций, описывающих правую часть, начальные и граничные условия
# для этого происходит считывание строк с соответствующими формулами из файла с заданием
# после чего эти строки парсятся с помощью библиотеки sympy, а результат ставится в соответствие лямбда-функции
t = Symbol("t")
x = Symbol("x")
# Начальные условия
ICS_parsed = parse_expr(task["ICS"])
ICS = lambdify(x, ICS_parsed)
# правая часть уравнения теплопроводности
f_parsed = parse_expr(task["rightPart"])
f = lambdify([t,x], f_parsed)
# граничные условия
beta0_parsed = parse_expr(task["BC"][0])
beta1_parsed = parse_expr(task["BC"][1])
beta0 = lambdify(t, beta0_parsed )
beta1 = lambdify(t, beta1_parsed )

# задание параметров сетки по t и x (число узлов)
# размер шагов сетки задан в файле (переменные tau и h)
# х меняется от х0 до х1
# t считаем меняется от 0 до 1
h = task["solver"]["h"]
tau = task["solver"]["tau"]
x1 = task["x1"]
x0 = task["x0"]
n_x = int((x1-x0)/h)
n_t = int(1 / tau)

# формирование матрицы СЛАУ
# В сеточном анадлоге система система дифференциальных уравнений заменяется на СЛАУ
# для неявной схемы соответствующая система описана в теоретической части отчета
a = task["a"]
p = -(a**2) * tau / (h ** 2)
q = 1 + 2 * (a**2) * tau / (h ** 2)
r = p
H = np.zeros((n_x + 1, n_x + 1))

for i in range(1, H.shape[0] - 1):
    H[i, i-1] = p
    H[i, i] = q
    H[i, i+1] = r
H[0, 0] = 1
H[n_x, n_x] = 1

U_t_x = np.zeros((n_t , n_x + 1))

# задание начального сечения температуры (tau=0)
U_t_x[0] = np.array([ICS(k*h) for k in range(n_x + 1)])

#решение уравнения теплопроводности - соответствующей СЛАУ для всех рассматриваемых моментов времени
t_i = 0
for j in range(1, n_t ):
    b_0_vec = np.zeros((n_x + 1, 1))
    b_1_vec = np.zeros((n_x + 1, 1))
    f_vec = np.zeros((n_x + 1, 1))
    b_0_vec[0, 0] = beta0(t_i)
    b_0_vec[n_x, 0] = beta1(t_i)
    t_i += tau
    x_i = 0
    for i in range(1, n_x):
        f_vec[i, 0] = tau * f(t_i, x_i)
        x_i += h
    b_1_vec[0, 0] = beta0(t_i)
    b_1_vec[n_x, 0] = beta1(t_i)
    U_t_x[j] = (np.dot(inv(H),((U_t_x[j - 1]).reshape((n_x + 1, 1)) + b_1_vec - b_0_vec + f_vec))).reshape((n_x + 1))
# визуализация полученных результатов
# выбирается какой-либо момент времени t0 (пусть t0= 0)
# строится двумерный график сечения температуры в этот момент времени ( т.е. строится зависимоть (x, u(x,t0))
u_print = U_t_x[0]
plt.plot(np.linspace(0,1,n_x+1),u_print)
plt.show()
