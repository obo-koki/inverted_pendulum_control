# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

class CartPole:
    """
    倒立振子モデル
    """
    g = 9.80665

    def __init__(self, m1, m2, l):
        self.m1 = m1
        self.m2 = m2
        self.l = l

    def state_equation(self, x, F):
        '''
        非線形モデル
        '''
        p = x[0]
        theta = x[1]
        dp = x[2]
        dtheta = x[3]

        dx = np.zeros(4)

        dx[0] = dp
        dx[1] = dtheta
        dx[2] = (-self.l*self.m2*np.sin(theta)*dtheta**2 + CartPole.g*self.m2*np.sin(2*theta)/2 + F)/(self.m1 + self.m2*np.sin(theta)**2)
        dx[3] = (CartPole.g*(self.m1 + self.m2)*np.sin(theta) - (self.l*self.m2*np.sin(theta)*dtheta**2 - F)*np.cos(theta))/(self.l*(self.m1 + self.m2*np.sin(theta)**2))

        return dx

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

        return A, B

def lqr(A, B, Q, R):
    '''
    最適レギュレータ計算
    '''
    P = linalg.solve_continuous_are(A, B, Q, R)
    K = linalg.inv(R).dot(B.T).dot(P)
    E = linalg.eigvals(A - B.dot(K))

    return P, K, E

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
    ip = CartPole(m1 = 1.0, m2 = 0.1, l = 0.8)

    # 最適レギュレータ計算
    A, B = ip.model_matrix()
    Q = np.diag([1, 100, 1, 10])
    R = np.eye(1)

    P, K, E = lqr(A, B, Q, R)
    print (K)

    # シミュレーション用変数初期化
    T = 10
    dt = 0.05
    x0 = np.array([0, 0.1, 0, 0]) * np.random.randn(1)

    t = np.arange(0, T, dt)
    x = np.zeros([len(t), 4])
    u = np.zeros([len(t), 1])

    x[0,:] = x0
    u[0] = 0

    # シミュレーションループ
    for i in range(1, len(t)):
        u[i] = -np.dot(K, x[i-1,:]) + np.random.randn(1)
        print ("i:",i,"u[i]:",u[i])
        dx = ip.state_equation(x[i-1,:], u[i])
        x[i,:] = x[i-1,:] + dx * dt


    # 時系列データプロット(x,u)
    plt.close('all')

    lbls = (r'$p$ [m]', r'$\theta$ [deg]', r'$\dot{p}$ [m/s]', r'$\dot{\theta}$ [deg/s]')
    scls = (1, 180/np.pi, 1, 180/np.pi)
    plot_graph(t, x, lbls, scls)

    lbls = (r'$F$ [N]',)
    scls = (1,)
    plot_graph(t, u, lbls, scls)

    # アニメーション表示
    fig, ax = plt.subplots()
    for i in range(len(t)):
        draw_pendulum(ax, t[i], x[i,0], x[i,1], ip.l)
        plt.pause(0.01)

if __name__ == '__main__':
    main()