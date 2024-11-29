import matplotlib.pyplot as plt
import numpy as np


def H(x):  # x - proportion of 1's in M
    return (1 - x) * np.log2(1 / (1 - x)) + x * np.log2(1 / x)


def misclassified_cnt(M, H_value):
    left = 0
    right = 0.5
    while right - left > 1e-10:
        if H_value > H((right - left) / 2):
            left += (right - left) / 2
        else:
            right -= (right - left) / 2

    return round(left * M)


x = np.linspace(1e-10, 1 - 1e-10, 100)  # avoid 0 and 1 to not divide by 0
y = H(x)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('H')
plt.title('y = (1 - x) * log(1/(1 - x)) + x * log(1/x)')
plt.show()

# 2
print(f'For M = 273 and H = 0.37, the number of misclassified instances is {misclassified_cnt(273, 0.37)}')
print(f'For M = 57 and H = 0.57, the number of misclassified instances is {misclassified_cnt(57, 0.57)}')
