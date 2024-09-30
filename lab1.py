from matplotlib import pyplot as plt
from scipy.stats import kurtosis, skew
import numpy as np
from scipy.stats import beta
from typing import Union
from scipy.optimize import minimize

def emp_distribution_func(x: np.ndarray, selection: np.ndarray):
    f_arr = []
    size = len(selection)
    for i in x:
        freq = (len(selection[selection < i])) / size
        f_arr.append(freq)
    return np.array(f_arr)

def emp_distr(x: np.ndarray):
    return emp_distribution_func(x, numbers)

def theoretic_beta(x: Union[float, np.ndarray], params):
    return beta.cdf(x, params[0], params[1])

def chi2_value_big(cdf2check, borders, params, nums, logging: bool = False):
    N = len(nums)
    if logging:
        print(f"borders {borders}")
        print(f"sample size: {N}")
    res = 0
    for i in range(len(borders)-1):
        p_k = cdf2check(borders[i+1], params) - cdf2check(borders[i], params)
        v_k = len([num for num in nums if borders[i] < num and num < borders[i+1]])
        if logging:
            print(f"curr borders: {borders[i], borders[i+1]}")
            print(f"v_k: {v_k}, p_k: {p_k}")
        try:
            res += (v_k - N*p_k)**2 / (N*p_k)
        except Exception:
            print(f"potential zero = {(N*p_k)}")

    return res


# читаем данные
data = open("NUMBER_6.txt")
numbers = data.read().split()

for i in range(len(numbers)):
    num = numbers[i].split('e')
    numbers[i] = float(num[0]) * 10 ** int(num[1])

data.close()

numbers = np.array(numbers)
numbers.sort()

# Выборочные характеристики
print(f"Выборочное среднее: {round(np.mean(numbers), 3)}")
print(f"Выборочная дисперсия: {round(np.var(numbers, ddof=1), 3)}")
print(f"Выб. коэф. асимметрии: {round(skew(numbers, bias=False), 3)}")
print(f"Выб. коэф. эксцесса: {round(kurtosis(numbers, bias=False), 3) - 3}")


#Гистограмма и ЭФР
left = min(numbers)-0.1
right = max(numbers)+0.1
x = np.linspace(left, right, 500)

f, ax = plt.subplots(1, 2, figsize=(9, 4))

ax[0].hist(numbers, density=True, bins=3) # изменила количество столбцов, иначе провалы
ax[0].grid()
ax[0].set_title('Гистограмма')

ax[1].plot(x, emp_distr(x))
ax[1].grid()
ax[1].set_title('ЭФР')

f.tight_layout()
plt.show()


# Доверительные полосы
epsilon = lambda gamma: np.sqrt(np.log(1 / (1 - gamma)) / 2 / len(numbers))
L = lambda gamma: np.array([max(F - epsilon(gamma), 0) for F in emp_distr(x)])
R = lambda gamma: np.array([min(F + epsilon(gamma), 1) for F in emp_distr(x)])

f, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(x, emp_distr(x), color='b', label='ЭФР')

for _, vargamma, clr in zip(range(2), [0.9, 0.95], ['r', 'g']):
    ax.plot(x, L(vargamma), color=clr, label=r"$\gamma$" + f" = {vargamma}")
    ax.plot(x, R(vargamma), color=clr)

ax.grid()
ax.legend()
plt.show()


#Проверка гипотезы о виде распределения - бета
print(numbers)
b = list(np.linspace(0.58, 0.87, 5))
b.insert(0, min(numbers))
b.extend([max(numbers)])
b.sort()

chi_2 = 5.99
borders = b
theta_beta = []
chi2 = lambda x: chi2_value_big(theoretic_beta, borders, x, nums=numbers, logging=False)
result = minimize(chi2, np.array([0.5, 0.1]), method='TNC', tol=1e-15)
print(f"Значение: {result['fun']}, min: {result['x']}")
theta_beta.append(result['x'])
print(result['fun'] < chi_2)

a_mle, b_mle, _, _ = beta.fit(numbers, floc=0)
print(a_mle, b_mle)

f, ax = plt.subplots(1, 2, figsize=(11, 6))

ax[0].hist(numbers, density=True, label='Гистограмма', bins=b)
ax[0].plot(x, beta.pdf(x, theta_beta[0][0], theta_beta[0][1]), label='Beta pdf')
ax[0].grid()
ax[0].legend()
ax[0].set_title('Гистограмма и ТФР')

ax[1].plot(x, emp_distr(x), label='ЭФР')
for _, vargamma, clr in zip(range(2), [0.9, 0.95], ['r', 'g']):
    ax[1].plot(x, L(vargamma), color=clr, label=r"$\gamma$" + f" = {vargamma}")
    ax[1].plot(x, R(vargamma), color=clr)
ax[1].grid()
ax[1].set_title('ЭФР и ТФР')
ax[1].plot(x, beta.cdf(x, theta_beta[0][0], theta_beta[0][1]), label='Beta cdf')
ax[1].legend()
plt.show()
print(a_mle, b_mle)

mean, var, skew, kurt = beta.stats(a_mle, b_mle, moments='mvsk')
print(f"    Среднее = {mean}\n    Дисперсия = {var}\n    Кф асимметрии = {skew}\n    Эксцесс = {kurt}")


