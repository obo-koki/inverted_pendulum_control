# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

class InvertedPendulum:
    #倒立振子モデル 参照"倒立２輪ロボットの安定化制御" 佐藤光 木澤悟
    g = 9.80665

    def __init__(self, m, M, L, r, I, J, D_fai, D_th, J_m, tau_A, tau, n, R_a, e_b, K_T, K_E):
        #ロボット本体パラメータ
        self.m = m #本体の質量
        self.M = M #車輪の質量
        self.L = L #車軸から測定した本体の重心距離
        self.r = r #車輪の半径
        self.I = I #本体の慣性モーメント
        self.J = J #車輪の慣性モーメント
        self.D_fai = D_fai #車輪の粘性摩擦係数
        self.D_th = D_th #本体の粘性摩擦係数
        #モータパラメータ
        self.J_m = J_m #モータの慣性モーメント
        self.tau_A = tau_A #モータ発生トルク
        self.tau = tau #モータ発生トルク
        self.n = n #モータの減速比
        self.R_a = R_a #電機子抵抗
        self.e_b = e_b #逆起電力
        self.K_T = K_T # トルク定数
        self.K_E = K_E # 誘起電圧係数
        #計算用の定数
        self.a = (m + M)*r**2 + J
        self.b = M * r * L
        self.c = M * L**2 + I
        self.d = M * InvertedPendulum.g * L
        self.E = self.a + J_m * self.n**2
        self.F = self.a + self.b
        self.G = D_th + (K_T * K_E * n**2)/R_a
        self.H = self.a + 2*self.b + self.c
        """
        self.a21 = -(self.E * self.d)/(self.F**2-self.E*self.H)
        self.a22 = (self.E*self.D_th)/(self.F**2-self.E*self.H)
        self.a23 = -(self.F*self.G)/(self.F**2-self.E*self.H)
        self.a31 = (self.F*self.d)/(self.F**2-self.E*self.H)
        self.a32 = (self.F*self.D_th)/(self.F**2-self.E*self.H)
        self.a33 = (self.G*self.H)/(self.F**2-self.E*self.H)
        self.b2 = (self.E*self.K_T*self.n)/(self.R_a*(self.F**2-self.E*self.H))
        self.b3 = -(self.H*self.K_T*self.n)/(self.R_a*(self.F**2-self.E*self.H))
        """
        self.a21 = 131.91
        self.a22 = -1.27 *10**(-4)
        self.a23 = 50.27
        self.a31 = -168.29
        self.a32 = 1.63 *10**(-4)
        self.a33 = -136.55
        self.b2 = -56.15
        self.b3 = 152.52
        #初期化
        self.xt = 0

    def state_equation(self, x, V):
        #非線形モデル（シミュレーション用）
        theta = x[0]
        dtheta = x[1]
        dfai = x[2]

        A = self.a + self.b * np.cos(theta)
        B = self.a + self.n**2 * self.J_m
        C = self.a + 2 * self.b * np.cos(theta) + self.c

        dx = np.zeros(3)
        dx[0] = dtheta
        dx[1] = self.a21 * theta + self.a22 * dtheta + self.a23 * dfai + self.b2 * V
        dx[2] = self.a31 * theta + self.a32 * dtheta + self.a33 * dfai + self.b3 * V
        #dx[1] = (B/(B*C-A**2))*(dtheta**2*self.b*np.sin(theta) + self.d*np.sin(theta)-self.D_th*dtheta-A/B*(dtheta**2*self.m*self.r*self.L*np.sin(theta)-(self.D_fai+(self.n**2*self.K_E*self.K_T)/self.R_a)*dfai+(self.n*self.K_T)/self.R_a*V))
        #dx[2] = (A/(A**2-B*C))*(dtheta**2*self.b*np.sin(theta) + self.d*np.sin(theta)-self.D_th*dtheta-C/A*(dtheta**2*self.m*self.r*self.L*np.sin(theta)-(self.D_fai+(self.n**2*self.K_E*self.K_T)/self.R_a)*dfai+(self.n*self.K_T)/self.R_a*V))
        return dx

    def model_matrix(self):
        #線形モデル（制御モデル用）
        A = np.array([ 
                [0, 1, 0],
                [self.a21, self.a22, self.a23],
                [self.a31, self.a32, self.a33]
            ])
        B = np.array([
                [0],
                [self.b2],
                [self.b3]
            ])

        return A, B

    def draw_pendulum(self, ax, t, theta, dtheta, dfai):
        #倒立振子プロット

        radius = self.r
        l = self.L
        self.xt += np.pi * (dtheta*dfai) * radius * 0.05#dt

        angles = np.arange(0.0, np.pi * 2.0, np.radians(3.0))
        ox = radius * np.cos(angles)
        oy = radius * np.sin(angles)

        bx = np.array([0.0, l * np.sin(-theta)]) + self.xt
        by = np.array([0, l * np.cos(-theta)]) + radius * 1.0

        wx = ox + self.xt
        wy = oy

        ax.cla()
        ax.plot(bx, by, "-k")
        ax.plot(wx, wy, "-k")
        ax.axis("equal")
        ax.set_ylim([-radius, radius+l])
        ax.set_title("t:%5.2f x:%5.2f theta:%5.2f" % (t, self.xt, theta))

def lqr(A, B, Q, R):
    '''
    最適レギュレータ計算
    '''
    P = linalg.solve_discrete_are(A, B, Q, R)
    K = linalg.inv(R).dot(B.T).dot(P)
    E = linalg.eigvals(A - B.dot(K))

    return P, K, E

def plot_graph(t, data, lbls, scls):
    #時系列プロット

    fig = plt.figure()

    nrow = int(np.ceil(data.shape[1] / 2))
    ncol = min(data.shape[1], 2)

    for i in range(data.shape[1]):
        ax = fig.add_subplot(nrow, ncol, i + 1)
        ax.plot(t, data[:,i] * scls[i])

        ax.set_xlabel('Time [s]')
        ax.set_ylabel(lbls[i])
        ax.grid()
        ax.set_xlim(t[0], t[-1])

    fig.tight_layout()


def main():
    # モデル初期化
    ip = InvertedPendulum(1,1,1,1,
        1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1)
    print (ip.J_m)
    print (ip.a21,ip.a22,ip.a23,ip.a31,ip.a32,ip.a33,ip.b2,ip.b3)
        #本体の質量
        #車輪の質量
        #車軸から測定した本体の重心距離
        #車輪の半径
        #本体の慣性モーメント
        #車輪の慣性モーメント
        #車輪の粘性摩擦係数
        #本体の粘性摩擦係数
        #モータの慣性モーメント
        #モータ発生トルク
        #モータ発生トルク
        #モータの減速比
        #電機子抵抗
        #逆起電力
        #トルク定数
        #誘起電圧係数

    # 最適レギュレータ計算
    A, B = ip.model_matrix()
    Q = np.diag([1, 1, 1])
    R = np.eye(1)

    _, K, _ = lqr(A, B, Q, R)
    print ("K:", K)

    # シミュレーション用変数初期化
    T = 3
    dt = 0.05
    x0 = np.array([0.1, 0, 0]) * np.random.randn(1)

    t = np.arange(0, T, dt)
    x = np.zeros([len(t), 3])
    u = np.zeros([len(t), 1])

    x[0,:] = x0
    u[0] = 0

    # シミュレーションループ
    for i in range(1, len(t)):
        u[i] = np.dot(K, x[i-1,:]) #+ np.random.randn(1)
        print ("i:",i,"u[i]:",u[i])
        dx = ip.state_equation(x[i-1,:], u[i])
        x[i,:] = x[i-1,:] + dx * dt
        while (x[i,0] > np.pi or x[i,0] < -np.pi):
            if x[i,0] > np.pi:
                x[i,0] -= 2*np.pi
            elif x[i,0] < -np.pi:
                x[i,0] += 2*np.pi
        print ("x[i]:",x[i,:])

    # 時系列データプロット(x,u)
    plt.close('all')

    lbls = (r'$p$ [m]', r'$\theta$ [deg]', r'$\dot{p}$ [m/s]', r'$\dot{\theta}$ [deg/s]')
    scls = (1, 180/np.pi, 1, 180/np.pi)
    plot_graph(t, x, lbls, scls)

    lbls = (r'$V$ [V]',)
    scls = (1,)
    plot_graph(t, u, lbls, scls)

    # アニメーション表示
    fig, ax = plt.subplots()
    for i in range(len(t)):
        ip.draw_pendulum(ax, t[i], x[i,0], x[i,1], x[i,2])
        plt.pause(0.01)

if __name__ == '__main__':
    main()