from control.matlab import *
import matplotlib.pyplot as plt
import numpy as np
A = [[0, 1],[-4, -5]]
B = [[0], [1]]
C = np.eye(2)
D = np.zeros([2, 1])
P = ss(A, B, C, D)
Td = np.arange(0, 5, 0.01)
x, t = step(P, Td,[1,0])
fig, ax = plt.subplots(figsize=(3, 2.3))
ax.plot(t, x[:,0], label = '$x_1$')
ax.plot(t, x[:,1], ls = '-.', label = '$x_2$')
ax.set_xticks(np.linspace(0, 5, 6))
ax.set_yticks(np.linspace(-0.4, 0.6, 6))
ax.legend()
ax.grid(ls=':')
plt.show()