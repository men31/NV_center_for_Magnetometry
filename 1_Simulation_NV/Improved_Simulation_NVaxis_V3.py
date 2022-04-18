from joblib import Parallel
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def flip_gaussian(x, cen, wid=1e6):
    return 1 - (np.exp(-(x - cen) ** 2 / 2 * wid) / (np.sqrt(2 * np.pi * wid)))


def Cutoff(x):
    return 1 if x > 0.8 else 0


class SimNV:
    def __init__(self):
        # NV axis
        u1 = np.array([[1], [1], [1]]) / 3 ** (1 / 2)
        u2 = np.array([[1], [-1], [1]]) / 3 ** (1 / 2)
        u3 = np.array([[-1], [1], [1]]) / 3 ** (1 / 2)
        u4 = np.array([[1], [1], [-1]]) / 3 ** (1 / 2)
        u = [u1, u2, u3, u4]
        self.u_dict = {f'NV_{i}': u[i] for i in range(len(u))}

        # Spin operator
        self.Sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
        self.Sy = np.array([[0, -1.j, 0], [1.j, 0, -1.j], [0, 1.j, 0]]) / np.sqrt(2)
        self.Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]]) 
        self.S = np.array((self.Sx, self.Sy, self.Sz))

        # NV properties
        self.gam = 28.8  # gamma NV = 28.8 GHz / T
        self.E = 5 / 1000  # E = 5 MHz
        self.D = 2.87  # D = 2.87 GHz

        # Initial properties
        self.theta = 0 * np.pi / 180
        self.phi = 0 * np.pi / 180
        self.B = 6 * 0.001  # B = 6 mT
        self.B_vec = self.B * np.array([[np.sin(self.theta) * np.cos(self.phi)],
                                        [np.sin(self.theta) * np.sin(self.phi)],
                                        [np.cos(self.theta)]])

        # Addition method
        self.cutoff = np.vectorize(Cutoff)
        self.project = True
        self.num = int(1e4)
        self.freX_min, self.freX_max = 2.6, 3.2 # GHz
        self.freX_lst = np.linspace(self.freX_min, self.freX_max, self.num)
        self.BX_min, self.BX_max = 0.0, 50.0 # mT
        self.BX_lst = np.linspace(self.BX_min, self.BX_max, self.num) / 1000
        self.NV_axis = None # For PL / Fre
        self.Fre_transition = None # For Fre / B
        self.Transition = None # For Tramsition / B

    ''' Core Program used to Calculate Eigenvalues of NV '''

    @staticmethod
    def Projection_B2U(B_vec, u_vec):
        P_u = u_vec @ u_vec.T / (u_vec.T @ u_vec)
        return P_u @ B_vec

    # Non-vectorize function 
    def Find_diffFre(self, B_vec, u):
        B_u = B_vec.copy()
        if self.project:
            B_u = self.Projection_B2U(B_vec, u)
        H_mag = (self.gam * np.dot(self.S.T, B_u)).reshape(3, 3)
        # H_zero = self.D * (self.Sz.T @ self.Sz) + self.E * ((self.Sx.T @ self.Sx) - (self.Sy.T @ self.Sy))
        H_zero = self.D * (self.Sz.T @ self.Sz)
        eigenval, eigenvec = np.linalg.eig(H_zero + H_mag)
        return eigenval, eigenvec

    # Vectorize function
    def Find_Multi_diffFre(self, B_vec, u):
        B_u = B_vec.copy()
        B_u = B_u.T.reshape(self.num, 3, 1)
        if self.project:
            B_u = self.Projection_B2U(B_u, u)
        H_mag = self.gam * np.dot(self.S.T, B_u).T.reshape(self.num, 3, 3)
        # H_zero = self.D * (self.Sz.T @ self.Sz) + self.E * ((self.Sx.T @ self.Sx) - (self.Sy.T @ self.Sy))
        H_zero = self.D * (self.Sz.T @ self.Sz) 
        eigenval, eigenvec = np.linalg.eig(H_zero + H_mag) 
        return eigenval, eigenvec

    ''' End Core Program '''

    ''' -------- CALCULATIION SECTIONS -------- '''

    ''' Method 0 : for Calculating Photoluminescence depened on (B, theta, phi) '''
    def A_PL_Fre(self, B, theta, phi, u):
        B_vec = B * np.array([[np.sin(theta) * np.cos(phi)], [np.sin(theta) * np.sin(phi)],
                                         [np.cos(theta)]])
        eigenVal, eigenVec = self.Find_diffFre(B_vec, u)
        max_idx = np.argmax(self.cutoff(abs(eigenVec)), 0)
        sort_eigenVal = eigenVal[max_idx]
        PL_minus = flip_gaussian(self.freX_lst, sort_eigenVal[0])
        PL_plus = flip_gaussian(self.freX_lst, sort_eigenVal[2])
        return PL_minus, PL_plus

    ''' Method 1 : for Calculating a Frequency for each spin (+1, -1, 0) on (B, theta, phi) '''
    def A_Fre_B(self, theta, phi, u):
        B_vec = self.BX_lst * np.array([[np.sin(theta) * np.cos(phi)], [np.sin(theta) * np.sin(phi)],
                                            [np.cos(theta)]])
        eigenVal, eigenVec = self.Find_Multi_diffFre(B_vec, u)
        max_idx = np.argmax(self.cutoff(abs(eigenVec)), 2)
        Frequency = np.take_along_axis(eigenVal, max_idx, axis=1)
        return Frequency[:, 0], Frequency[:, 1], Frequency[:, 2]

    ''' Method 2 : for Calculating a Transition Frequency for each spin (+1, -1, 0) on (B, theta, phi) '''
    def A_Transition_B(self, theta, phi, u):
        B_vec = self.BX_lst * np.array([[np.sin(theta) * np.cos(phi)], [np.sin(theta) * np.sin(phi)],
                                            [np.cos(theta)]])
        eigenVal, eigenVec = self.Find_Multi_diffFre(B_vec, u)
        max_idx = np.argmax(self.cutoff(abs(eigenVec)), 2)
        Frequency = np.take_along_axis(eigenVal, max_idx, axis=1)
        MsPlus, MsZero, MsMinus = Frequency[:, 0], Frequency[:, 1], Frequency[:, 2]
        # print(MsPlus[:5] - MsZero[:5])
        return MsPlus - MsZero, MsMinus - MsZero
        

    ''' -------- END CALCULATIION SECTIONS -------- '''

    ''' -------- PLOT & USER INTERFACE SECTIONS -------- '''

    ''' Plot Photoluminescence from Method 1 in Form of Gaussian Function'''
    def Plot_Gaussian(self):
        self.NV_axis = {}
        fig, ax = plt.subplots()
        for key in self.u_dict:
            PL_minus, PL_plus = self.A_PL_Fre(self.B, self.theta, self.phi, self.u_dict[key])
            line_minus, = plt.plot(self.freX_lst, PL_minus, lw=2, label=key + '_-')
            line_plus, = plt.plot(self.freX_lst, PL_plus, lw=2, label=key + '_+')
            self.NV_axis[key + '_-'] = line_minus
            self.NV_axis[key + '_+'] = line_plus
        plt.subplots_adjust(left=0.15, bottom=0.355, top=0.906, right=0.799)
        plt.title('NV axis')
        plt.xlabel('Frequency (Ghz)')
        plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
        plt.ylabel('PL')

        ax.margins(x=0)
        axcolor = 'lightgoldenrodyellow'
        axB = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=axcolor)
        axTheta = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        axPhi = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

        self.sB = Slider(axB, 'B (mT)', 0.1, 50, valinit=self.B)
        self.sTheta = Slider(axTheta, 'Theta (degree)', 0.0, 180.0, valinit=self.theta, valstep = 0.5)
        self.sPhi = Slider(axPhi, 'Phi (degree)', 0.0, 360.0, valinit=self.phi, valstep = 0.5)
        def update(val):
            B = self.sB.val * 0.001
            theta = self.sTheta.val * np.pi /180
            phi = self.sPhi.val * np.pi / 180
            for key in self.u_dict:
                PL_minus, PL_plus = self.A_PL_Fre(B, theta, phi, self.u_dict[key])
                self.NV_axis[key + '_-'].set_ydata(PL_minus)
                self.NV_axis[key + '_+'].set_ydata(PL_plus)
            fig.canvas.draw_idle()

        self.sB.on_changed(update)
        self.sTheta.on_changed(update)
        self.sPhi.on_changed(update)
        
        def reset(event):
            self.sB.reset()
            self.sTheta.reset()
            self.sPhi.reset()
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        button.on_clicked(reset)

        plt.show()

    ''' Plot Frequency of (+1, -1, 0) Spin Versus Magnetic field (B) '''
    def Plot_Fre_B(self):
        self.Fre_transition = {}
        fig, ax = plt.subplots()
        for key in self.u_dict:
            Fre_plus, Fre_g, Fre_minus = self.A_Fre_B(self.theta, self.phi, self.u_dict[key])
            line_minus, = plt.plot(self.BX_lst*1e3, Fre_minus.reshape(-1), '.--', label=key+ '_-')
            line_plus, = plt.plot(self.BX_lst*1e3, Fre_plus.reshape(-1), '--', label=key+ '_+')
            line_g, = plt.plot(self.BX_lst*1e3, Fre_g.reshape(-1), '-', label=key+ '_0')
            self.Fre_transition[key+'_-'] = line_minus
            self.Fre_transition[key+'_+'] = line_plus
            self.Fre_transition[key+'_0'] = line_g
        plt.subplots_adjust(left=0.15, bottom=0.355, top=0.906, right=0.799)
        plt.title('NV axis')
        plt.xlabel('Magnetic Field B (mT)')
        plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
        plt.ylabel('Frequency (Ghz)')

        ax.margins(x=0)
        axcolor = 'lightgoldenrodyellow'
        axTheta = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        axPhi = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

        sTheta = Slider(axTheta, 'Theta (degree)', 0.0, 180.0, valinit=self.theta, valstep = 0.5)
        sPhi = Slider(axPhi, 'Phi (degree)', 0.0, 360.0, valinit=self.phi, valstep = 0.5)
        def update(val):
            theta = sTheta.val * np.pi /180
            phi = sPhi.val * np.pi / 180
            for key in self.u_dict:
                Fre_plus, Fre_g, Fre_minus = self.A_Fre_B(theta, phi, self.u_dict[key])
                self.Fre_transition[key+'_-'].set_ydata(Fre_minus.reshape(-1))
                self.Fre_transition[key+'_+'].set_ydata(Fre_plus.reshape(-1))
                self.Fre_transition[key+'_0'].set_ydata(Fre_g.reshape(-1))
            fig.canvas.draw_idle()

        sTheta.on_changed(update)
        sPhi.on_changed(update)
        
        def reset(event):
            sTheta.reset()
            sPhi.reset()
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        button.on_clicked(reset)
        plt.show()

    ''' Plot Transition Frequency of (+1, -1, 0) Spin Versus Magnetic field (B) '''
    def Plot_Transition_B(self):
        self.Transition = {}
        fig, ax = plt.subplots()
        for key in self.u_dict:
            Trans_plus, Trans_minus = self.A_Transition_B(self.theta, self.phi, self.u_dict[key])
            line_minus, = plt.plot(self.BX_lst*1e3, Trans_minus.reshape(-1), '.--', label=key+ '_-')
            line_plus, = plt.plot(self.BX_lst*1e3, Trans_plus.reshape(-1), '--', label=key+ '_+')
            self.Transition[key+'_-'] = line_minus
            self.Transition[key+'_+'] = line_plus
        plt.subplots_adjust(left=0.15, bottom=0.355, top=0.906, right=0.799)
        plt.title('NV axis')
        plt.xlabel('Magnetic Field B (mT)')
        plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
        plt.ylabel('Transition Frequency (Ghz)')

        ax.margins(x=0)
        axcolor = 'lightgoldenrodyellow'
        axTheta = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        axPhi = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

        sTheta = Slider(axTheta, 'Theta (degree)', 0.0, 180.0, valinit=self.theta, valstep = 0.5)
        sPhi = Slider(axPhi, 'Phi (degree)', 0.0, 360.0, valinit=self.phi, valstep = 0.5)
        def update(val):
            theta = sTheta.val * np.pi /180
            phi = sPhi.val * np.pi / 180
            for key in self.u_dict:
                Trans_plus, Trans_minus = self.A_Transition_B(theta, phi, self.u_dict[key])
                self.Transition[key+'_-'].set_ydata(Trans_minus.reshape(-1))
                self.Transition[key+'_+'].set_ydata(Trans_plus.reshape(-1))
            fig.canvas.draw_idle()

        sTheta.on_changed(update)
        sPhi.on_changed(update)
        
        def reset(event):
            sTheta.reset()
            sPhi.reset()
        resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
        button.on_clicked(reset)
        plt.show()

    ''' -------- END PLOT & USER INTERFACE SECTIONS -------- '''

    ''' Main program '''
    def Calculate(self, method='default'):
        if method == 'default' or method == 'pl-fre' or method == 0:
            self.Plot_Gaussian()
        elif method == 'fre-mag' or method == 1:
            self.Plot_Fre_B()
        elif method == 'trans-mag' or method == 2:
            self.Plot_Transition_B()
        else:
            print('[-] ERROR !!! ')
            print('[-] HAVE NOT THIS METHOD   !!! ')
            print('[-] PLEASE ENTER METHOD = [0/1/2]')