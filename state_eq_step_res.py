from control.matlab import *
import matplotlib.pyplot as plt
import numpy as np

A = [[0, 1],[-4, -5]]
B = [[0], [1]]
C = np.eye(2)
D = np.zeros([2, 1])
P = ss(A, B, C, D)
Td = np.arange(0, 5, 0.01)
Ud = 1*(Td>0) #ステップ入力
X0 = [-0.3, 0.4]
xst, t = step(P, Td) #ゼロ状態応答（ステップ入力）
xin, _ = initial(P, Td, X0) #ゼロ入力応答
x, _, _ = lsim(P, Ud, Td, X0)
fig, ax = plt.subplots(1, 2, figsize=(6, 2.3))
for i in [0, 1]:
    ax[i].plot(t, x[:,i], label='response')
    ax[i].plot(t, xst[:,i], ls='--', label='zero state')
    ax[i].plot(t, xin[:,i], ls='-.', label='zero input')
    ax[i].grid(ls=':')
ax[0].set_xlabel('t')
ax[0].set_ylabel('$x_1$')
ax[1].set_xlabel('t')
ax[1].set_ylabel('$x_2$')
ax[1].legend()
fig.tight_layout()

plt.show()