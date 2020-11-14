import numpy as np
from control.matlab import *
import matplotlib.pyplot as plt

A = [[0, 1],[-4, 5]]
B = [[0], [1]]
C = np.eye(2)
D = np.zeros([2, 1])
P = ss(A, B, C, D)
print (P)
Pole = [-0.5, -1]
F = -acker(P.A, P.B, Pole)
Acl = P.A + P.B*F
Pfb = ss(Acl, P.B, P.C, P.D)
Td = np.arange(0, 5, 0.01)
X0 = [-0.6, 0.4]
x, t = initial(Pfb, Td, X0) #ゼロ入力応答
fig, ax = plt.subplots(figsize=(3, 2.3))
ax.plot(t, x[:,0], label = '$x_1$')
ax.plot(t, x[:,1], ls = '-.', label = '$x_2$')
ax.grid(ls=':')
plt.show()