import numpy as np
import matplotlib.pyplot as plt

# functions
def f1(n):
    return 210 * n + 210

def f2(n):
    return n ** 3.5 - 1000

def f3(n):
    return 100 * n ** 2.1 + 50

# ranges for x-axis
n_values_5 = np.arange(0, 5, 0.1)
n_values_15 = np.arange(0, 15, 0.1)
n_values_50 = np.arange(0, 50, 0.1)

# Plot for n = 5
plt.figure(figsize=(8, 6))
plt.plot(n_values_5, f1(n_values_5), 'r', label="f1(n) = 210n + 210")
plt.xlabel("n")
plt.ylabel("f1(n) = 210n + 210")
plt.title("Function Plot for f1(n) = 210n + 210")
plt.legend()
plt.show()

# n = 15
plt.figure(figsize=(8, 6))
plt.plot(n_values_15, f2(n_values_15), 'b', label="f2(n) = n^3.5 - 1000")
plt.xlabel("n")
plt.ylabel("f2(n) = n^3.5 - 1000")
plt.title("Function Plot for f2(n) = n^3.5 - 1000")
plt.legend()
plt.show()

# n = 50
plt.figure(figsize=(8, 6))
plt.plot(n_values_50, f3(n_values_50), 'g', label="f3(n) = 100n^2.1 + 50")
plt.xlabel("n")
plt.ylabel("f3(n) = 100n^2.1 + 50")
plt.title("Function Plot for f3(n) = 100n^2.1 + 50")
plt.legend()
plt.show()