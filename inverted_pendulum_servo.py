# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

class InvertedPendulum:
    #倒立振子モデル 参照"倒立２輪ロボットの安定化制御" 佐藤光 木澤悟
    g = 9.80665

    def __init__(self, m=None, M=None, L=None, r=None, I=None, J=None, 
        D_fai=None, D_th=None, J_m=None, n=None, R_a=None, e_b=None, 
        K_T=None, K_E=None):

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
        self.n = n #モータの減速比
        self.R_a = R_a #電機子抵抗
        self.e_b = e_b #逆起電力
        self.K_T = K_T # トルク定数
        self.K_E = K_E # 誘起電圧係数
        #初期化
        self.xt = 0
    
    def set_robot_param(self, m, M, L, r, I, J, D_fai=0, D_th=0):
        self.m = m #本体の質量
        self.M = M #車輪の質量
        self.L = L #車軸から測定した本体の重心距離
        self.r = r #車輪の半径
        self.I = I #本体の慣性モーメント
        self.J = J #車輪の慣性モーメント
        self.D_fai = D_fai #車輪の粘性摩擦係数
        self.D_th = D_th #本体の粘性摩擦係数
    
    def set_motor_param(self,J_m, n, R_a, e_b, K_T, K_E):
        self.J_m = J_m #モータの慣性モーメント
        self.n = n #モータの減速比
        self.R_a = R_a #電機子抵抗
        self.e_b = e_b #逆起電力
        self.K_T = K_T # トルク定数
        self.K_E = K_E # 誘起電圧係数

    def calc_control_model(self):
        #線形モデル（制御モデル用）->拡大系
        #計算用の定数
        self.a = (self.m + self.M)*self.r**2 + self.J
        self.b = self.M * self.r * self.L
        self.c = self.M * self.L**2 + self.I
        self.d = self.M * InvertedPendulum.g * self.L
        self.E = self.a + self.J_m * self.n**2
        self.F = self.a + self.b
        self.G = self.D_th + (self.K_T * self.K_E * self.n**2)/self.R_a
        self.H = self.a + 2*self.b + self.c

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
        """

        self.A = np.array([ 
                [0, 1, 0],
                [self.a21, self.a22, self.a23],
                [self.a31, self.a32, self.a33]
            ])
        self.B = np.array([
                [0],
                [self.b2],
                [self.b3]
            ])
        
        self.C = np.array([
                [0, 0, 1]
            ])
        """
        self.C = np.array([
                [0, 1, 1]
            ]) * self.r
        """

        #拡大系
        AB = np.append(self.A, self.B, axis=1)
        self.Ce = np.append(self.C, np.zeros((1,1)), axis=1)
        self.Ae = np.append(AB,np.zeros((1,4)),axis=0)
        self.Be = np.array([
            [0],
            [0],
            [0],
            [1]
        ])

    def lqr(self, Q, R):
        #
        Pe = linalg.solve_continuous_are(self.Ae, self.Be, Q, R)
        Ke = R * self.Be.T.dot(Pe)
        AB = self.Ae[0:3,0:4]
        ABC0 = np.append(AB, self.Ce, axis=0)
        self.K = Ke.dot(linalg.inv(ABC0))
        print ("K:",self.K)

    def state_equation(self, x, V):
        #非線形モデル（シミュレーション用）
        theta = x[0]
        dtheta = x[1]
        dfai = x[2]

        A = self.a + self.b * np.cos(theta)
        B = self.a + self.n**2 * self.J_m
        C = self.a + 2 * self.b * np.cos(theta) + self.c
        D = (self.n * self.K_T * V) / self.R_a

        dx = np.zeros(3)
        dx[0] = dtheta
        #dx[1] = self.a21 * theta + self.a22 * dtheta + self.a23 * dfai + self.b2 * V
        #dx[2] = self.a31 * theta + self.a32 * dtheta + self.a33 * dfai + self.b3 * V
        #dx[1] = (B/(B*C-A**2))*(dtheta**2*self.b*np.sin(theta) + self.d*np.sin(theta)-self.D_th*dtheta-A/B*(dtheta**2*self.m*self.r*self.L*np.sin(theta)-(self.D_fai+(self.n**2*self.K_E*self.K_T)/self.R_a)*dfai+(self.n*self.K_T)/self.R_a*V))
        #dx[2] = (A/(A**2-B*C))*(dtheta**2*self.b*np.sin(theta) + self.d*np.sin(theta)-self.D_th*dtheta-C/A*(dtheta**2*self.m*self.r*self.L*np.sin(theta)-(self.D_fai+(self.n**2*self.K_E*self.K_T)/self.R_a)*dfai+(self.n*self.K_T)/self.R_a*V))

        dx[1] = ( self.m*self.r*self.L*np.sin(theta) - A/B*self.b*np.sin(theta) )*dtheta**2 - (self.D_fai + self.n*D)*dfai + D - A/B*self.d*np.sin(theta) + A/B*self.D_th*dtheta
        dx[1] = (B/(B**2-A*C))*dx[1] 
        dx[2] = ( C/B*self.m*self.r*self.L*np.sin(theta) - self.b*np.sin(theta) )*dtheta**2 - C/B*(self.D_fai+self.n*D)*dfai + C/B*D - self.d*np.sin(theta) + self.D_th*dtheta
        dx[2] = (B/(C*A-B**2))*dx[2]
        return dx
    
    def simulation(self, T, dt, x0, target_vel):
        # シミュレーション用変数初期化
        t = np.arange(0, T, dt)
        x = np.zeros([len(t), 3])
        u = np.zeros([len(t), 1])
        fai = np.zeros([len(t), 1])

        x[0,:] = x0
        u[0] = 0
        diff = 0
        fai[0] = 0

        # シミュレーションループ
        for i in range(1, len(t)):
            print ()
            diff += (target_vel - x[i-1,2]) *dt
            diff = 0
            print ("i:",i)
            print ("x[i-1]:",x[i-1,:],"diff:",diff)
            u[i] = -np.dot(self.K[:,0:3], x[i-1,:])+np.dot(self.K[0,3], diff) #+ np.random.randn(1)
            print ("u[i]:",u[i])
            dx = self.state_equation(x[i-1,:], u[i])
            x[i,:] = x[i-1,:] + dx * dt
            fai[i] = x[i-1,2] * dt
            if (x[i,0] > np.pi or x[i,0] < -np.pi):
                print("倒立振子転倒!")
                T = i * dt
                break

        # 時系列データプロット(x,u)
        plt.close('all')

        lbls = (r'$p$ [m]', r'$\theta$ [deg]', r'$\dot{p}$ [m/s]', r'$\dot{\theta}$ [deg/s]')
        scls = (1, 180/np.pi, 1, 180/np.pi)
        self.plot_graph(t, x, lbls, scls)

        lbls = (r'$V$ [V]',)
        scls = (1,)
        self.plot_graph(t, u, lbls, scls)

        # アニメーション表示
        _, ax = plt.subplots()
        for i in range(len(t)):
            self.draw_pendulum(ax, t[i], x[i,0], x[i,1], x[i,2], fai[i])
            plt.pause(dt)

    def draw_pendulum(self, ax, t, theta, dtheta, dfai, fai):
        #倒立振子プロット
        radius = self.r
        l = self.L
        self.xt += np.pi * (dtheta*dfai) * radius * 0.05#dt

        angles = np.arange(0.0, np.pi * 2.0, np.radians(3.0))
        ox = radius * np.cos(angles)
        oy = radius * np.sin(angles)

        #車体
        bx = np.array([0.0, l * np.sin(-theta)]) + self.xt
        by = np.array([0, l * np.cos(-theta)]) + radius * 1.0

        #車輪
        wx = ox + self.xt
        wy = oy
        
        #回転の目印
        px = radius * np.sin(fai) + self.xt
        py = radius * np.cos(fai)

        ax.cla()
        ax.plot(bx, by, "-k")
        ax.plot(wx, wy, "-k")
        ax.plot(px, py, marker='.', markersize=20)
        ax.axis("equal")
        ax.set_ylim([-1*radius, radius+l])
        ax.set_title("t:%5.2f x:%5.2f theta:%5.2f" % (t, self.xt, theta))

    def plot_graph(self, t, data, lbls, scls):
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
    ip = InvertedPendulum()
    ip.set_robot_param(1,1,1,1,1,1,1,1)
    #ip.set_robot_param(0.281,0.102,0.037,0.029,1.639**10*(-4),
    #    2.357*10**(-5),2.00*10**(-7),1.00*10**(-4))
    #ip.set_robot_param(0.641,0.0485,0.065,0.0575/2,0.02263,
    #    3.497e-05,0.001016,0.0009073)
        #本体の質量,車輪の質量, 車軸から測定した本体の重心距離, 車輪の半径, 本体の慣性モーメント
        #車輪の慣性モーメント, 車輪の粘性摩擦係数, 本体の粘性摩擦係数
    ip.set_motor_param(1,1,1,1,1,1)
    #ip.set_motor_param(5.208e-08,19*50/12, 10.4, 1, 0.006198, 0.008441)
        #モータの慣性モーメント, モータの減速比, 電機子抵抗, 逆起電力, トルク定数, 誘起電圧係数

    # 制御用のモデル導出(Ae, Be)
    ip.calc_control_model()

    # リカッチ方程式を解く
    Q = np.diag([1*10**(3), 1.0*10**(2), 1.0, 1.5])
    R = np.eye(1)
    #Q = np.diag([1*10**(8), 1.5*10**(8), 50, 15])
    #R = np.eye(1)
    ip.lqr(Q,R)

    #シミュレーション
    T = 5 #シミュレーション時間
    dt = 0.1 #刻み時間
    x0 = np.array([0.3, 0, 0]) #状態変数初期値 x=[th, th/dt, fai/dt]
    target_vel = 0.1 # m/s
    ip.simulation(T, dt, x0, target_vel)

if __name__ == '__main__':
    main()