# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from cart_pole_reglator import CartPole

class CartPoleServo(CartPole):
    def __init__(self, m1, m2, l):
        super().__init__(m1, m2, l)
    
    def model_matrix(self):
        '''
        線形モデル
        '''
        A = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, CartPole.g*self.m2/self.m1, 0, 0],
            [0, CartPole.g*(self.m1 + self.m2)/(self.l*self.m1), 0, 0]
        ])

        B = np.array([
                [0],
                [0],
                [1/self.m1],
                [1/(self.l*self.m1)]
            ])
        '''
        線形モデル（拡大系)
        xdot = [ A  0 ] x + [ B ] u
               [-C  0 ]     [ 0 ]
        '''
        C = np.array([0,1,0,0])
        #print (C)

        #A_Zero = np.zeros((4,4))

        #A0 = np.append(A, A_Zero, axis=1)
        #C0 = np.append(-C, A_Zero, axis=1)
        #A_bar = np.append(A0,C0,axis=0)
        #print (A_bar)

        #B_Zero = np.zeros((4,1))
        
        #B_bar = np.append(B,B_Zero,axis=0)
        #print (B_bar)

        return A, B, C

def lqr(A, B, C, Q, R):
    '''
    最適サーボ計算
    '''
    P = linalg.solve_continuous_are(A, B, Q, R)
    F_0 = -linalg.inv(R).dot(B.T).dot(P)
    X = A - B.dot(linalg.inv(R)).dot(B.T).dot(P)
    Y = -C.dot(linalg.inv(X)).dot(B)

    if Y.ndim ==1:
        if Y[0] == 0:
            H_0 = Y
        else:
            H_0 = np.array([1/Y[0]])
    else:
        H_0 = linalg.inv(Y)

    return F_0, H_0

def plot_graph(t, data, lbls, scls):
    '''
    時系列プロット
    '''

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

def draw_pendulum(ax, t, xt, theta, l):
    '''
    倒立振子プロット
    '''

    cart_w = 1.0
    cart_h = 0.4
    radius = 0.1

    cx = np.array([-0.5, 0.5, 0.5, -0.5, -0.5]) * cart_w + xt
    cy = np.array([0.0, 0.0, 1.0, 1.0, 0.0]) * cart_h + radius * 2.0

    bx = np.array([0.0, l * np.sin(-theta)]) + xt
    by = np.array([cart_h, l * np.cos(-theta) + cart_h]) + radius * 2.0

    angles = np.arange(0.0, np.pi * 2.0, np.radians(3.0))
    ox = radius * np.cos(angles)
    oy = radius * np.sin(angles)

    rwx = ox + cart_w / 4.0 + xt
    rwy = oy + radius
    lwx = ox - cart_w / 4.0 + xt
    lwy = oy + radius

    wx = ox + float(bx[1])
    wy = oy + float(by[1])

    ax.cla()
    ax.plot(cx, cy, "-b")
    ax.plot(bx, by, "-k")
    ax.plot(rwx, rwy, "-k")
    ax.plot(lwx, lwy, "-k")
    ax.plot(wx, wy, "-k")
    ax.axis("equal")
    ax.set_xlim([-cart_w, cart_w])
    ax.set_title("t:%5.2f x:%5.2f theta:%5.2f" % (t, xt, theta))

def main():
    # モデル初期化
    ip = CartPoleServo(m1 = 1.0, m2 = 0.1, l = 0.8)

    # 最適レギュレータ計算 K:フィードバックゲイン
    A, B, C = ip.model_matrix()
    Q = np.diag([1, 1000, 10, 1]) #決め方が分からない
    R = np.eye(1)
    F_0, H_0 = lqr(A, B, C, Q, R)
    print (F_0)
    print (H_0)

    # シミュレーション用変数初期化
    T = 10
    dt = 0.05
    x0 = np.array([0, 0, 0, 0]) * np.random.randn(1)

    # 目標値
    x_target = np.array([0])
    vel_target = np.array([0,0,0.5,0])

    w0 = x_target - x0

    t = np.arange(0, T, dt)
    x = np.zeros([len(t), 4])
    w = np.zeros([len(t), 4])
    u = np.zeros([len(t), 1])

    x[0,:] = x0
    w[0,:] = w0
    u[0] = 0

    # シミュレーションループ
    for i in range(1, len(t)):
        u[i] = F_0.dot(x[i-1,:]-vel_target) + H_0.dot(x_target)
        dx = ip.state_equation(x[i-1,:], u[i])
        x[i,:] = x[i-1,:] + dx * dt
        #w[i,:] = x_target - x[i-1,:] + w[i-1,:]


    # 時系列データプロット(x,u)
    plt.close('all')

    lbls = (r'$p$ [m]', r'$\theta$ [deg]', r'$\dot{p}$ [m/s]', r'$\dot{\theta}$ [deg/s]')
    scls = (1, 180/np.pi, 1, 180/np.pi)
    plot_graph(t, x, lbls, scls)

    lbls = (r'$F$ [N]',)
    scls = (1,)
    plot_graph(t, u, lbls, scls)

    # アニメーション表示
    _, ax = plt.subplots()
    for i in range(len(t)):
        draw_pendulum(ax, t[i], x[i,0], x[i,1], ip.l)
        plt.pause(0.01)

if __name__ == '__main__':
    main()